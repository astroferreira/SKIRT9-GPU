"""Cached, bounded volume reconstruction for browser-side light transport.

Only data preparation runs here. No SKIRT jobs, GPU rendering, or original
catalogue writes are performed. The browser integrates emission/extinction.
"""
import hashlib
import importlib.util
import itertools
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from astropy.cosmology import LambdaCDM
from pydantic import BaseModel, ConfigDict, Field
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve

from .catalog import RawDataset, neighbour_kernel_radii, stamp, table_chunks
from .physics import physical_options
from .preview_spectra import RESOURCE_ROOT, luminosities, mean_extinction, resource

VERSION = 2
EXPORTER = Path('/home/astroferrreira/src/illustris_to_stellarmap/physical_model.py')
MSUN_PER_KPC2 = 1.98847e30 / 3.085677581491367e19**2


@contextmanager
def exclusive_preparation(lock):
    if not lock.acquire(blocking=False):
        raise ValueError('Another preview volume is being prepared. Please retry Build / update shortly.')
    try:
        yield
    finally:
        lock.release()


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    red: str = 'SLOAN_SDSS_I'
    green: str = 'SLOAN_SDSS_R'
    blue: str = 'SLOAN_SDSS_G'
    size: Literal[64, 96, 128, 192, 256, 512] = 256
    span: float = Field(default=60, ge=.1, le=10000)
    x: float = Field(default=0, ge=-10000, le=10000)
    y: float = Field(default=0, ge=-10000, le=10000)
    z: float = Field(default=0, ge=-10000, le=10000)
    kernel: float = Field(default=.5, ge=.1, le=8)

    @property
    def bands(self):
        return (self.red, self.green, self.blue)

    @property
    def center(self):
        return np.array([self.x, self.y, self.z])


def cic(positions, weights, center, span, size):
    """Cloud-in-cell deposition on cell centres, zero outside the preview cube."""
    coordinates = (positions-center+span/2)*(size/span)-.5
    low = np.floor(coordinates).astype(np.int64)
    fraction = coordinates-low
    weights = np.asarray(weights).reshape(len(positions), -1)
    result = np.zeros((size, size, size, weights.shape[1]), dtype=np.float32)
    for corner in itertools.product((0, 1), repeat=3):
        indices = low+corner
        valid = np.all((indices >= 0) & (indices < size), axis=1)
        if not valid.any():
            continue
        factor = np.prod(np.where(corner, fraction, 1-fraction), axis=1)[valid]
        x, y, z = indices[valid].T
        flat = (z*size+y)*size+x  # WebGL textures use x-fastest order
        for channel in range(weights.shape[1]):
            result[..., channel] += np.bincount(flat, weights=weights[valid, channel]*factor,
                                                minlength=size**3).reshape(size, size, size).astype(np.float32)
    return result


def available_memory_bytes():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])*1024
    except (OSError, ValueError):
        pass
    return None


def check_reconstruction_memory(size, pad, channels, smoothed):
    # Conservative working-set estimate, including the seven output channels,
    # CIC temporaries, and padded real-FFT work arrays. Broad kernels can cost
    # much more than the final textures, especially for a 512-cubed request.
    required = size**3*(7+2*channels)*4 + (size+2*pad)**3*(4*channels+12)
    if smoothed:
        required += (2*pad+1)**3*16 + next_fast_len(size+4*pad, real=True)**3*32
    available = available_memory_bytes()
    if available is not None and required > available*.75:
        raise ValueError(f'{size}³ reconstruction needs an estimated {required/1024**3:.1f} GiB of working memory '
                         f'for this kernel group; {available/1024**3:.1f} GiB is available. '
                         'Choose a smaller volume grid or retry when more server RAM is free.')


def reconstruct(positions, radii, weights, request):
    """Multiscale cubic-spline convolution of luminosity-conserving CIC grids.

    Support radii are quantized in sqrt(2) steps; unresolved sources use CIC.
    This is an explicitly finite-resolution reconstruction, not a Voronoi RT grid.
    """
    n, dx = request.size, request.span/request.size
    weights = np.asarray(weights).reshape(len(positions), -1)
    result = np.zeros((n, n, n, weights.shape[1]), dtype=np.float32)
    if not len(positions):
        return result
    h = np.maximum(np.asarray(radii)*request.kernel/dx, 0)
    groups = np.maximum(0, np.rint(2*np.log2(np.maximum(h, 1))).astype(int))
    for group in np.unique(groups):
        selected = (groups == group) & np.all(np.abs(positions-request.center) < request.span/2+(h[:, None]+1)*dx, axis=1)
        if not selected.any():
            continue
        # Pad to retain kernels of sources just outside the requested cube.
        radius = 2.**(group/2)
        pad = min(n, int(np.ceil(radius))) if group > 0 else 1
        check_reconstruction_memory(n, pad, weights.shape[1], group > 0)
        grid = cic(positions[selected], weights[selected], request.center,
                   request.span+2*pad*dx, n+2*pad)
        if group > 0:
            axis = np.arange(-pad, pad+1, dtype=np.float32)
            zz, yy, xx = np.meshgrid(axis, axis, axis, indexing='ij', sparse=True)
            q = np.sqrt(xx*xx+yy*yy+zz*zz)/radius
            kernel = np.where(q < .5, 1-6*q*q+6*q**3, np.where(q < 1, 2*(1-q)**3, 0))
            kernel *= 8/(np.pi*radius**3)
            if radius <= pad:
                kernel /= kernel.sum()
            for channel in range(weights.shape[1]):
                grid[..., channel] = fftconvolve(grid[..., channel], kernel, mode='same')
        result += np.maximum(grid[pad:pad+n, pad:pad+n, pad:pad+n], 0)
    return result/dx**3


def source_name(node):
    if node.tag == 'BpassSEDFamily':
        return 'BpassSEDFamily_'+node.get('imf', 'Chabrier100')+('_downsampled' if node.get('resolution', 'Downsampled') == 'Downsampled' else '')
    if node.tag == 'BruzualCharlotSEDFamily':
        return 'BruzualCharlotSEDFamily_'+node.get('imf', 'Chabrier')+('_lr' if node.get('resolution', 'Low') == 'Low' else '_hr')
    if node.tag == 'ToddlersSEDFamily' and node.get('sedMode', 'SFRNormalized') == 'SFRNormalized':
        template = {'BPASSChab100Bin':'BPASS_chab100_bin', 'BPASSChab300Bin':'BPASS_chab300_bin', 'SB99Kroupa100Sin':'SB99_kroupa100_sin'}[node.get('stellarTemplate', 'BPASSChab100Bin')]
        return ('ToddlersSEDFamily_SFRNormalized_'+template+('_Dust' if node.get('includeDust', 'true') == 'true' else '_noDust')
                + ('_hr' if node.get('resolution', 'High') == 'High' else '_lr')
                + ('_30Myr' if node.get('sfrPeriod', 'Period10Myr') == 'Period30Myr' else '_10Myr'))
    raise ValueError(f'{node.tag} is not supported by the band preview; the diagnostic view remains available')


def prepared_sources(dataset):
    layers = []
    nodes = [node for node in dataset.tree().iter() if node.tag in ('ParticleSource', 'ParticleMedium', 'VoronoiMeshMedium')]
    for table, node in zip(dataset.tables(), nodes):
        kind, offset = table['kind'], table['velocity_columns']
        if kind == 'dust':
            if node.get('massType', 'Mass') != 'Mass' or node.get('importMetallicity', 'false') != 'false':
                raise ValueError('Band preview currently requires dust mass columns without metallicity scaling')
            radius_column = table['radius_column']
            columns = [0, 1, 2, table['weight_column']]+([radius_column] if radius_column is not None else [])
        else:
            family = next((child for child in node.iter() if child.tag.endswith('SEDFamily')), None)
            if family is None:
                raise ValueError('Source SED family missing')
            name = source_name(family)
            columns = [0, 1, 2, 3, *range(4+offset, (8 if kind == 'young' else 7)+offset)]
        chunks = list(table_chunks(table['path'], columns))
        if not chunks:
            continue
        values = np.concatenate(chunks)
        xyz = values[:, :3]/1000
        if kind == 'dust':
            radius = values[:, 4]/1000 if radius_column is not None else np.zeros(len(values))
            weight = values[:, 3]*float(node.get('massFraction', 1))
            parameters = np.empty((len(values), 0)); name = ''
        elif kind == 'stars':
            radius = neighbour_kernel_radii(xyz, xyz)
            weight, parameters = values[:, 4], values[:, 5:7]
        else:
            radius, weight, parameters = values[:, 3]/1000, values[:, 7], values[:, 4:7]
        if np.any(weight < 0) or np.any(radius < 0) or np.any(parameters < 0):
            raise ValueError('Negative source or dust properties')
        layers.append(dict(kind=kind, name=name, positions=xyz, radii=radius, parameters=parameters, weight=weight))
    return layers


def raw_sources(dataset, request):
    # Share the installed physical exporter calculations without invoking an
    # export, creating a job, or modifying its original files.
    spec = importlib.util.spec_from_file_location('observatory_physical_preview', EXPORTER)
    physics = importlib.util.module_from_spec(spec); spec.loader.exec_module(physics)
    with h5py.File(dataset.path) as f:
        hdr = f['Header'].attrs
        a, h, box = [float(hdr[k]) for k in ('Time', 'HubbleParam', 'BoxSize')]
        for key, expected in [('UnitLength_in_cm', 3.085678e21), ('UnitMass_in_g', 1.989e43), ('UnitVelocity_in_cm_per_s', 1e5)]:
            if key in hdr and not np.isclose(hdr[key], expected, rtol=1e-4):
                raise ValueError(f'Unsupported raw snapshot unit: {key}')
        center = (np.asarray(dataset.entry['center_native'])+request.center/(a/h)) % box
        # Extra spatial padding reduces source-kernel truncation at cube edges.
        region = request.span*1.5
        s, _ = physics.read_selected(f['PartType4'], center, a/h*1000, box, region*1000,
                                    ['GFM_StellarFormationTime', 'GFM_InitialMass', 'GFM_Metallicity'])
        formed = (s['GFM_StellarFormationTime'] > 0) & (s['GFM_StellarFormationTime'] <= a)
        s = {key:value[formed] for key, value in s.items()}
        cosmo = LambdaCDM(H0=100*h, Om0=float(hdr['Omega0']), Ode0=float(hdr['OmegaLambda']), Ob0=float(hdr.get('OmegaBaryon', .0486)), Tcmb0=2.73)
        ages = physics.stellar_ages(cosmo, s['GFM_StellarFormationTime'], a)
        radii = neighbour_kernel_radii(s['xyz'], s['xyz'])
        stars = np.column_stack((s['xyz'], radii, s['GFM_InitialMass']*1e10/h, s['GFM_Metallicity'], ages))
        gas = np.empty((0, 8))
        if 'PartType0' in f:
            fields = ['Masses', 'GFM_Metallicity', 'Density', 'StarFormationRate', 'InternalEnergy', 'ElectronAbundance', 'GFM_Metals']
            g, _ = physics.read_selected(f['PartType0'], center, a/h*1000, box, region*1000, fields)
            hydrogen = g['GFM_Metals'][:, 0] if 'GFM_Metals' in g else np.full(len(g['xyz']), .75)
            mu = 4/(1+3*hydrogen+4*hydrogen*g['ElectronAbundance'])
            temperature = (2/3)*g['InternalEnergy']*1e6*mu*1.67262192369e-27/1.380649e-23
            gas = np.column_stack((g['xyz'], g['Masses']*1e10/h, g['GFM_Metallicity'], g['Density']*10*h*h/a**3, g['StarFormationRate'], temperature))
    old, young, _ = physics.source_populations(stars, gas, .025, 320., 'clip-report', 'exclude')
    dust = physics.diffuse_dust(gas, h, physical_options(dataset.entry)['diffuse_dust'])
    return [dict(kind='stars', name='BpassSEDFamily_Chabrier100_downsampled', positions=old[:, :3]/1000+request.center,
                 radii=old[:, 3]/1000, weight=old[:, 4], parameters=old[:, 5:7]),
            dict(kind='young', name='ToddlersSEDFamily_SFRNormalized_BPASS_chab100_bin_Dust_hr_10Myr',
                 positions=young[:, :3]/1000+request.center, radii=young[:, 3]/1000, weight=young[:, 7], parameters=young[:, 4:7]),
            dict(kind='dust', name='', positions=dust[:, :3]/1000+request.center, radii=dust[:, 3]/1000,
                 weight=dust[:, 4], parameters=np.empty((len(dust), 0)))]


class PreviewVolumes:
    def __init__(self, catalog):
        self.catalog = catalog
        self.root = catalog.cache/'browser-volumes'
        self.lock = threading.Lock()

    def build(self, identifier, request):
        from .simulation import BANDS
        if any(band not in BANDS for band in request.bands):
            raise ValueError('Unknown preview filter')
        dataset = self.catalog.datasets[identifier]
        if dataset.entry.get('render_available') is False:
            raise ValueError('This large view-only catalogue is not enabled for volume preparation; use the diagnostic view')
        fingerprint, _ = dataset.fingerprint()
        resources = [stamp(p) for p in sorted(RESOURCE_ROOT.rglob('*.stab'))]
        identity = dict(version=VERSION, dataset=fingerprint, request=request.model_dump(), resources=resources,
                        exporter=stamp(EXPORTER) if isinstance(dataset, RawDataset) else None)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = self.root/key
        if (folder/'volume.json').is_file():
            return json.loads((folder/'volume.json').read_text())
        with exclusive_preparation(self.lock):
            if (folder/'volume.json').is_file():
                return json.loads((folder/'volume.json').read_text())
            folder.mkdir(parents=True, exist_ok=True)
            # Cache the full source properties independently of filters and grid
            # quality so subsequent colour/quality changes avoid catalogue scans.
            source_identity = {k:identity[k] for k in ('version', 'dataset', 'exporter')}
            if isinstance(dataset, RawDataset):
                source_identity['region'] = [request.x, request.y, request.z, request.span]
            source_key = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()
            source_file = self.root/f'sources-{source_key}.npz'
            if source_file.is_file():
                with np.load(source_file, allow_pickle=False) as archive:
                    layers = [{field:archive[f'{i}_{field}'] for field in ('positions', 'radii', 'weight', 'parameters')} |
                              {field:str(archive[f'{i}_{field}']) for field in ('kind', 'name')} for i in range(int(archive['count']))]
            else:
                layers = raw_sources(dataset, request) if isinstance(dataset, RawDataset) else prepared_sources(dataset)
                temporary = source_file.with_suffix('.partial.npz')
                np.savez(temporary, count=len(layers), **{f'{i}_{field}':value for i, layer in enumerate(layers) for field, value in layer.items()})
                os.replace(temporary, source_file)
            info = dataset.info()
            redshift = info['observer_redshift'] if not info['distance_mpc'] else 0.
            fields = np.zeros((request.size, request.size, request.size, 7), dtype=np.float32)
            counts, input_luminosity, input_dust = {}, np.zeros(3), 0.
            for layer in layers:
                kind = layer['kind']; counts[kind] = counts.get(kind, 0)+len(layer['positions'])
                if not len(layer['positions']):
                    continue
                if kind == 'dust':
                    weight = layer['weight'][:, None]
                    input_dust += float(weight.sum())
                else:
                    weight = luminosities(layer['name'], request.bands, redshift, layer['parameters'], layer['weight'])/1e36
                    input_luminosity += weight.sum(axis=0)
                reconstructed = reconstruct(layer['positions'], layer['radii'], weight, request)
                if kind == 'dust':
                    fields[..., 3] += reconstructed[..., 0]
                else:
                    fields[..., :3] += reconstructed if kind == 'stars' else 0
                    fields[..., 4:7] += reconstructed if kind == 'young' else 0
            dx = request.span/request.size
            retained_light = (fields[..., :3].sum(axis=(0, 1, 2), dtype=float)+fields[..., 4:7].sum(axis=(0, 1, 2), dtype=float))*dx**3
            retained_dust = float(fields[..., 3].sum(dtype=float)*dx**3)
            normalization = max(float(retained_light.max())/request.span**2/25, 1e-30)
            fields[..., :3] /= normalization; fields[..., 4:7] /= normalization
            density_scale = max(float(fields[..., 3].max()), 1.)
            fields[..., 3] /= density_scale
            # One common RGB scale preserves channel ratios. Keep float32 on
            # disk/GPU to retain the astronomical emissivity dynamic range.
            for name, data in [('stars-dust.bin', fields[..., :4]), ('young.bin', np.concatenate((fields[..., 4:7], np.zeros_like(fields[..., :1])), axis=-1))]:
                data.astype('<f4').tofile(folder/name)
            warnings = ['Finite-resolution cubic-spline reconstruction (sqrt(2)-binned support radii); unresolved detail is limited by voxel size.',
                        'Mean Milky Way extinction averaged over each filter, not the configured multigrain opacity or a wavelength-resolved band transfer.',
                        'Direct source emission plus extinction only: no scattered light or diffuse heated-dust/PAH emission.',
                        'Broadband lookup interpolation is approximate; preview values are relative radiance, not calibrated FITS photometry.',
                        'The displayed cube is cropped; rebuild around the camera to inspect another region.']
            if not isinstance(dataset, RawDataset):
                warnings.append('Dust masses are regridded from the full catalogue; Voronoi cell geometry is approximated.')
            else:
                warnings.append('Raw snapshot sources use the physical exporter BPASS/TODDLERS and diffuse-dust prescriptions; 32-neighbour kernels use all valid stars within a 1.5x padded region.')
            meta = dict(key=key, dataset_id=identifier, size=request.size, bounds_kpc=[(request.center-request.span/2).tolist(), (request.center+request.span/2).tolist()],
                        voxel_kpc=dx, bands=list(request.bands), redshift=redshift, counts=counts,
                        extinction_per_kpc=(np.asarray(mean_extinction(request.bands, redshift))*MSUN_PER_KPC2*density_scale).tolist(),
                        retained_dust_msun=retained_dust, input_dust_msun=input_dust,
                        input_luminosity_scaled=input_luminosity.tolist(), retained_luminosity_scaled=retained_light.tolist(),
                        luminosity_unit='1e36 W, neutral band luminosity (observer pivot times mean rest L_lambda)', display_normalization=normalization,
                        warnings=warnings, request=request.model_dump(), dtype='float32-le', channels=4,
                        files={name:f'/api/preview-volumes/{key}/{name}' for name in ('stars-dust.bin', 'young.bin')})
            temporary = folder/'volume.partial.json'; temporary.write_text(json.dumps(meta, indent=2)); os.replace(temporary, folder/'volume.json')
            return meta
