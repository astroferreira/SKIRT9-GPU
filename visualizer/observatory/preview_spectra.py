"""Read-only SKIRT resource tables and compact broadband preview lookups.

The layout and interpolation flags follow SKIRT/core/StoredTable{,Impl}.
Integrating at population grid nodes before interpolating is a preview
approximation; this module never replaces SKIRT's spectral calculation.
"""
import itertools
import os
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

RESOURCE_ROOT = Path(os.environ.get('SKIRT_PREVIEW_RESOURCES', Path(__file__).resolve().parents[3] / 'resources'))


def resource(name):
    matches = sorted(RESOURCE_ROOT.rglob(name + '.stab'))
    if len(matches) != 1:
        raise ValueError(f'Preview needs exactly one SKIRT resource: {name}')
    return matches[0]


class StoredTable:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as stream:
            def integer():
                return struct.unpack('<Q', stream.read(8))[0]
            def strings(n):
                return [stream.read(8).decode('ascii').strip() for _ in range(n)]
            if stream.read(8) != b'SKIRT X\n' or integer() != 0x010203040A0BFEFF:
                raise ValueError('Invalid SKIRT stored table')
            n = integer()
            if not 1 <= n <= 6:
                raise ValueError('Unsupported stored-table dimensions')
            self.names, self.units, self.scales = strings(n), strings(n), strings(n)
            self.axes = [np.fromfile(stream, dtype='<f8', count=integer()) for _ in range(n)]
            if any(len(a) < 2 or not np.isfinite(a).all() for a in self.axes):
                raise ValueError('Invalid stored-table axes')
            nq = integer()
            self.quantities, self.quantity_units, self.quantity_scales = strings(nq), strings(nq), strings(nq)
            # Quantity index changes fastest, followed by wavelength, Z, age, ...
            self.values = np.memmap(self.path, dtype='<f8', mode='r', offset=stream.tell(),
                                    shape=(nq, *[len(a) for a in self.axes]), order='F')

    def quantity(self, name):
        return self.values[self.quantities.index(name)]


def interpolate(axes, scales, values, parameters, logarithmic=True):
    """Clamped multilinear interpolation; trailing dimensions are channels."""
    parameters = np.asarray(parameters, dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError('Non-finite population parameters')
    indices, fractions = [], []
    for i, (axis, scale) in enumerate(zip(axes, scales)):
        if np.any(np.diff(axis) <= 0):
            raise ValueError('Population lookup axes must be strictly increasing')
        p = np.clip(parameters[:, i], axis[0], axis[-1])
        grid = np.log(axis) if scale == 'log' else axis
        if scale == 'log':
            p = np.log(p)
        low = np.clip(np.searchsorted(grid, p, side='right') - 1, 0, len(grid) - 2)
        indices.append(low)
        fractions.append((p - grid[low]) / (grid[low+1] - grid[low]))
    result = np.zeros((len(parameters), *values.shape[len(axes):]))
    zero = np.zeros_like(result, dtype=bool)
    for corner in itertools.product((0, 1), repeat=len(axes)):
        weight = np.ones(len(parameters))
        for bit, fraction in zip(corner, fractions):
            weight *= fraction if bit else 1-fraction
        value = values[tuple(index+bit for index, bit in zip(indices, corner))]
        weight = weight.reshape((len(parameters),) + (1,)*(result.ndim-1))
        if logarithmic:
            zero |= (weight > 0) & (value <= 0)
            value = np.log(np.maximum(value, 1e-300))
        result += weight*value
    if logarithmic:
        result = np.where(zero, 0, np.exp(result))
    return result


@lru_cache(maxsize=12)
def band_lut(name, bands, redshift, signature):
    table = StoredTable(resource(name))
    wave = table.axes[0]
    spectra = table.quantity('Llambda')
    channels = []
    for band in bands:
        transmission = StoredTable(resource(band + '_BroadBand'))
        # Include every native template and filter knot in the band integral.
        bw = transmission.axes[0]
        observed = np.unique(np.concatenate((bw, wave*(1+redshift))))
        observed = observed[(observed >= bw[0]) & (observed <= bw[-1])]
        rest = observed/(1+redshift)
        weights = np.interp(observed, bw, transmission.quantity('T'))
        lower = np.clip(np.searchsorted(wave, rest, side='right')-1, 0, len(wave)-2)
        x, a, b = rest, wave[lower], wave[lower+1]
        if table.scales[0] == 'log':
            x, a, b = np.log(x), np.log(a), np.log(b)
        fraction = ((x-a)/(b-a)).reshape((-1,) + (1,)*(spectra.ndim-1))
        v0, v1 = spectra[lower], spectra[lower+1]
        if table.quantity_scales[table.quantities.index('Llambda')] == 'log':
            sampled = np.exp((1-fraction)*np.log(np.maximum(v0, 1e-300)) + fraction*np.log(np.maximum(v1, 1e-300)))
            sampled = np.where((v0 > 0) & (v1 > 0), sampled, 0)
        else:
            sampled = (1-fraction)*v0 + fraction*v1
        sampled[(rest < wave[0]) | (rest > wave[-1])] = 0
        # Mean L_lambda through the normalized SKIRT filter. A common redshift
        # flux factor cancels in this relative preview; no per-channel scaling.
        integral = np.trapezoid(weights, observed)
        pivot = np.sqrt(integral / np.trapezoid(weights/observed**2, observed))
        # Observatory FITS use Neutral flux units: lambda_pivot * <L_lambda>.
        channels.append(pivot*np.trapezoid(sampled*weights.reshape((-1,) + (1,)*(sampled.ndim-1)), observed, axis=0)
                        / integral)
    return table.axes[1:], table.scales[1:], np.stack(channels, axis=-1)


def luminosities(name, bands, redshift, parameters, mass):
    paths = [resource(name), *[resource(b+'_BroadBand') for b in bands]]
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    axes, scales, lut = band_lut(name, tuple(bands), redshift, signature)
    values = interpolate(axes, scales, lut, parameters)
    return values*np.asarray(mass)[:, None]


def mean_extinction(bands, redshift):
    """Filter-averaged mean Milky Way extinction, m^2/kg of dust.

    Deliberately labeled an approximation, not the configured multigrain mix.
    """
    table = StoredTable(resource('MeanInterstellarOpticalProps'))
    kappa = (table.quantity('sigmaabs') + table.quantity('sigmasca')) / table.quantity('mu')
    # The distributed mean curve has two out-of-order EUV edge knots.
    order = np.argsort(table.axes[0], kind='stable')
    result = []
    for band in bands:
        curve = StoredTable(resource(band+'_BroadBand'))
        wave, weight = curve.axes[0], curve.quantity('T')
        opacity = np.exp(np.interp(np.log(wave/(1+redshift)), np.log(table.axes[0][order]), np.log(kappa[order])))
        result.append(float(np.trapezoid(opacity*weight, wave)/np.trapezoid(weight, wave)))
    return result
