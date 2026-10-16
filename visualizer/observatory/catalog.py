"""Read the exact SKIRT input catalogues; cache a bounded uniform display sample."""

import hashlib
import json
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import h5py
from astropy.cosmology import LambdaCDM
from scipy.spatial import cKDTree
from .physics import physical_options, dust_description


KERNEL_NEIGHBOURS = 32
CLOUD_FORMAT_VERSION = 3


def display_strength(weight):
    """Compress a positive luminosity/mass proxy without discarding contrast."""
    weight = np.asarray(weight, dtype=float)
    positive = weight[np.isfinite(weight) & (weight > 0)]
    if not len(positive):
        return np.ones(len(weight))
    reference = np.median(positive)
    return np.clip(np.sqrt(np.maximum(weight, 0) / reference), .12, 8.)


def stellar_colors(age_yr):
    """Continuous, display-oriented stellar-age palette from blue to amber/red."""
    log_age = np.log10(np.clip(np.asarray(age_yr, dtype=float), 1e6, 1.5e10))
    stops = np.array([6., 7., 8., 9., np.log10(1.5e10)])
    palette = np.array([[.34, .80, 1.00], [.44, .65, 1.00], [.76, .61, .98],
                        [1.00, .73, .36], [1.00, .34, .16]])
    return np.column_stack([np.interp(log_age, stops, palette[:, channel]) for channel in range(3)])


def stellar_strength(mass, age_yr):
    """Broad bolometric-like preview proxy; not a replacement for an SED lookup."""
    age = np.clip(np.asarray(age_yr, dtype=float), 1e6, 1.5e10)
    light_proxy = np.asarray(mass, dtype=float) * (age / 1e9) ** -.7
    return display_strength(light_proxy)


def neighbour_kernel_radii(query, reference, neighbours=KERNEL_NEIGHBOURS):
    """Radius to the Nth nearest *other* source, evaluated against the full layer."""
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if not len(query):
        return np.empty(0)
    if len(reference) <= 1:
        return np.full(len(query), 1e-3)
    # Query includes the source itself at zero distance, hence N+1 samples.
    k = min(len(reference), neighbours + 1)
    distances = cKDTree(reference).query(query, k=k, workers=-1)[0]
    radii = distances if distances.ndim == 1 else distances[:, -1]
    positive = radii[np.isfinite(radii) & (radii > 0)]
    fallback = np.median(positive) if len(positive) else 1e-3
    return np.maximum(np.nan_to_num(radii, nan=fallback, posinf=fallback), np.finfo(float).eps)


def write_cloud_layer(path, positions, colors, radii, strength):
    values = np.column_stack((positions, np.clip(colors, 0, 1), radii, strength)).astype("<f4")
    values.tofile(path)


def length_kpc(value):
    parts = value.split()
    factors = {"pc": .001, "kpc": 1., "Mpc": 1000., "m": 1 / 3.085677581491367e19}
    if len(parts) != 2 or parts[1] not in factors:
        raise ValueError(f"An explicit supported length unit is required: {value}")
    return float(parts[0]) * factors[parts[1]]


def stamp(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def table_chunks(path, columns):
    # Exporter tables are plain text with pc positions. Fail on unsupported units
    # rather than silently moving the catalogue relative to the SKIRT instrument.
    with path.open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            if "column" in line.lower() and any(f"column {i}:" in line.lower() for i in (1, 2, 3)):
                if "(" in line and "(pc)" not in line:
                    raise ValueError(f"Viewer requires pc position columns in {path.name}")
    with path.open() as stream:
        while True:
            lines = []
            for line in stream:
                if line.strip() and not line.lstrip().startswith("#"):
                    lines.append(line)
                if len(lines) >= 50000:
                    break
            if not lines:
                break
            values = np.loadtxt(lines, usecols=columns, ndmin=2)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite catalogue values in {path.name}")
            yield values


class Dataset:
    def __init__(self, entry, root):
        self.entry = entry
        self.id = entry["id"]
        self.path = (root / entry["base_ski"]).resolve()

    def tree(self):
        return ET.parse(self.path)

    def tables(self):
        result = []
        for element in self.tree().iter():
            if element.tag not in ("ParticleSource", "VoronoiMeshMedium", "ParticleMedium"):
                continue
            if element.get("useColumns", ""):
                raise ValueError("Custom useColumns mappings need an explicit viewer adapter")
            kind = "dust" if "Medium" in element.tag else (
                "young" if any(element.find(".//" + tag) is not None for tag in ("MappingsSEDFamily", "ToddlersSEDFamily")) else "stars")
            if kind == "stars" and not any(element.find(".//" + tag) is not None for tag in ("BruzualCharlotSEDFamily", "BpassSEDFamily")):
                raise ValueError("Unsupported stellar SED family")
            velocity_columns = 3 if element.get("importVelocity", "false") == "true" else 0
            if velocity_columns and element.tag != 'ParticleSource':
                raise ValueError("Moving diffuse media need an explicit viewer adapter")
            if element.get('importVelocityDispersion', 'false') == 'true':
                raise ValueError('Unresolved source dispersion needs an explicit viewer adapter')
            path = (self.path.parent / element.attrib["filename"]).resolve()
            legacy_dust = self.entry.get("legacy_dust_columns")
            weight_column = (legacy_dust[-1] if legacy_dust else 3) if element.tag == "VoronoiMeshMedium" else (4 if kind == "dust" else (7 if element.find(".//ToddlersSEDFamily") is not None else 4))
            if kind != 'dust': weight_column += velocity_columns
            radius_column = (3 if legacy_dust and legacy_dust[-1] != 3 else None) if element.tag == "VoronoiMeshMedium" else 3
            result.append({"kind": kind, "path": path, "tag": element.tag,
                           "weight_column": weight_column, "radius_column": radius_column,
                           "velocity_columns": velocity_columns})
        if not result:
            raise ValueError("No supported particle or dust catalogues found")
        return result

    def fingerprint(self):
        inputs = [stamp(t["path"]) for t in self.tables()]
        payload = {"base_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                   "inputs": inputs, "adapter": self.entry.get("legacy_dust_columns"),
                   "version": CLOUD_FORMAT_VERSION, "physics": self.entry.get("physics")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload

    def info(self):
        base = {k: self.entry[k] for k in ("id", "name", "description", "notes", "physics", "render_available", "render_unavailable_reason") if k in self.entry}
        try:
            root = self.tree().getroot()
            frame = root.find(".//FrameInstrument")
            if frame is None:
                raise ValueError("No FrameInstrument in model")
            for table in self.tables():
                if not table["path"].is_file():
                    raise ValueError(f"Missing catalogue: {table['path'].name}")
            cosmology = root.find(".//FlatUniverseCosmology")
            dust = root.find(".//DustEmissionOptions")
            base.update(available=True, fov_kpc=length_kpc(frame.attrib["fieldOfViewX"]),
                        observer_redshift=float(cosmology.get("redshift", 0)) if cosmology is not None else 0,
                        distance_mpc=length_kpc(frame.get("distance", "10 Mpc")) / 1000,
                        dust_emission=dust.get("dustEmissionType", "Equilibrium") if dust is not None else "None",
                        simulation_mode=root.find(".//MonteCarloSimulation").get("simulationMode"),
                        base_ski=str(self.path))
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            base.update(available=False, error=str(error))
        return base


class RawDataset:
    """One central MOOSE galaxy backed directly by an AREPO snapshot."""
    def __init__(self, entry):
        self.entry=entry;self.id=entry['id'];self.path=Path(entry['raw_snapshot']).resolve()

    def fingerprint(self):
        payload={'raw_snapshot':stamp(self.path),'group_catalog':stamp(Path(self.entry['group_catalog'])),
                 'center_native':self.entry['center_native'],'preview_fov_kpc':self.entry['preview_fov_kpc'],
                 'version':CLOUD_FORMAT_VERSION,'physics':self.entry['physics'],'physical_options':physical_options(self.entry)}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

    def info(self):
        base={k:self.entry[k] for k in ('id','name','description','notes','physics')}
        try:
            if not self.path.is_file() or not Path(self.entry['group_catalog']).is_file():raise ValueError('Raw snapshot or group catalogue is missing')
            base.update(available=True,raw=True,fov_kpc=self.entry['preview_fov_kpc'],
                        default_model_fov_kpc=self.entry['default_model_fov_kpc'],observer_redshift=self.entry['redshift'],
                        distance_mpc=0,dust_emission='Stochastic',simulation_mode='DustEmission',base_ski=None)
        except (OSError,ValueError) as error:base.update(available=False,error=str(error),raw=True)
        return base


class Catalog:
    def __init__(self, config, cache, max_points=100000):
        document=json.loads(config.read_text());entries=list(document["datasets"])
        for series in document.get('raw_series',[]):entries.extend(self.expand_raw_series(series))
        self.datasets = {e["id"]: (RawDataset(e) if e.get('raw_snapshot') else Dataset(e,config.parent)) for e in entries}
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_points = max_points
        self.lock = threading.Lock()

    @staticmethod
    def expand_raw_series(series):
        output=Path(series['output_dir']).resolve();result=[]
        options = physical_options(series)
        for snapshot in sorted(output.glob('snap_*.hdf5')):
            number=int(snapshot.stem.split('_')[-1]);groups=output/f'fof_subhalo_tab_{number:03d}.hdf5'
            if not groups.is_file():continue
            with h5py.File(snapshot) as raw,h5py.File(groups) as catalog:
                header=raw['Header'].attrs;first=int(catalog['Group']['GroupFirstSub'][0])
                if first < 0 or first >= len(catalog['Subhalo']['SubhaloPos']):
                    continue
                center=catalog['Subhalo']['SubhaloPos'][first].astype(float).tolist()
                redshift=float(header['Redshift']);a=float(header['Time']);h=float(header['HubbleParam']);box=float(header['BoxSize'])
            result.append(dict(id=f"{series['id_prefix']}-{number:03d}",name=f"{series['name']} · snapshot {number:03d}",
                description=f"{series['name']} · raw AREPO snapshot {number:03d} · primary central subhalo {first} · z={redshift:.3f}",
                notes='Raw HDF5 preview. Stars and gas are sampled directly; dust is a metal-mass proxy. Physical BPASS/TODDLERS catalogues are generated and cached only when an observation is submitted.',
                raw_snapshot=str(snapshot),group_catalog=str(groups),snapshot=number,subhalo=first,center_native=center,
                preview_fov_kpc=series.get('preview_fov_kpc',300),default_model_fov_kpc=series.get('default_model_fov_kpc',300),
                redshift=redshift,scale_factor=a,hubble_param=h,box_size_native=box,
                physical_options=dict(options),
                physics=dict(model=f"on-demand-bpass-toddlers-{options['diffuse_dust']}-{options['diffuse_grain_mix']}-v1",snapshot_redshift=redshift,source_model='BPASS + TODDLERS generated on demand',
                    diffuse_dust=dust_description(options),selection='Primary central subhalo from GroupFirstSub[0]',
                    accounting='Empirical diffuse + unresolved clouds; combined dust mass unknown')))
        return result

    def cloud(self, identifier):
        dataset = self.datasets[identifier]
        if isinstance(dataset,RawDataset):return self.raw_cloud(dataset)
        digest, provenance = dataset.fingerprint()
        cache_dir = self.cache / f"{identifier}-{digest[:16]}-{self.max_points}"
        with self.lock:
            if (cache_dir / "cloud.json").is_file():
                return cache_dir, json.loads((cache_dir / "cloud.json").read_text())
            cache_dir.mkdir(parents=True, exist_ok=True)
            layers = []
            all_positions = []
            for index, table in enumerate(dataset.tables()):
                rng = np.random.default_rng(12345 + index)
                kind = table["kind"]
                if kind == "stars":
                    offset = table['velocity_columns']
                    columns = (0, 1, 2, 4+offset, 6+offset)  # position, initial mass, age
                else:
                    columns = (0, 1, 2, table["weight_column"])
                    if table["radius_column"] is not None and table["radius_column"] not in columns:
                        columns += (table["radius_column"],)
                kept = np.empty((0, len(columns)))
                keys = np.empty(0)
                reference_positions = []
                count = 0
                positive_count = 0
                total_weight = 0.
                lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
                for values in table_chunks(table["path"], columns):
                    count += len(values)
                    if kind == "stars":
                        if np.any(values[:, 3:] < 0):
                            raise ValueError("Negative stellar mass or age in preview catalogue")
                    else:
                        if np.any(values[:, 3] < 0):
                            raise ValueError("Negative preview mass/SFR")
                        total_weight += float(values[:, 3].sum())
                        values = values[values[:, 3] > 0]
                    positive_count += len(values)
                    if not len(values):
                        continue
                    if kind == "stars":
                        reference_positions.append(values[:, :3] / 1000)
                    lo = np.minimum(lo, values[:, :3].min(axis=0))
                    hi = np.maximum(hi, values[:, :3].max(axis=0))
                    candidate = np.concatenate((kept, values))
                    ranks = np.concatenate((keys, rng.random(len(values))))
                    indices = np.argpartition(ranks, self.max_points - 1)[:self.max_points] if len(ranks) > self.max_points else np.arange(len(ranks))
                    kept, keys = candidate[indices], ranks[indices]
                if not positive_count:
                    continue
                positions = kept[:, :3] / 1000
                all_positions.append(positions)
                if kind == "stars":
                    colors = stellar_colors(kept[:, 4])
                    strength = stellar_strength(kept[:, 3], kept[:, 4])
                    radii = neighbour_kernel_radii(positions, np.concatenate(reference_positions))
                    kernel = {"profile": "SKIRT projected cubic spline", "neighbours": KERNEL_NEIGHBOURS,
                              "radius": "distance to 32nd nearest other source in full catalogue"}
                else:
                    colors = np.tile([.40, .84, 1.] if kind == "young" else [.46, .34, .22], (len(kept), 1))
                    strength = display_strength(kept[:, 3])
                    radii = np.maximum(kept[:, 4] / 1000, np.finfo(float).eps) if len(columns) == 5 else np.zeros(len(kept))
                    kernel = ({"profile": "SKIRT projected cubic spline", "radius": "catalogue smoothing length"}
                              if len(columns) == 5 else {"profile": "screen-space point", "radius": "not available for Voronoi sites"})
                name = f"{index}-{kind}.bin"
                write_cloud_layer(cache_dir / name, positions, colors, radii, strength)
                layers.append({"kind": kind, "count": count, "display_count": len(kept), "file": name,
                               "bounds_kpc": [(lo / 1000).tolist(), (hi / 1000).tolist()],
                               "positive_count": positive_count, "weight_total": total_weight if kind != "stars" else None,
                               "weight_unit": "Msun" if kind == "dust" else "Msun/yr" if kind == "young" else None,
                               "kernel": kernel, "kernel_h_median_kpc": float(np.median(radii)) if np.any(radii) else None})
            if not layers:
                raise ValueError("The catalogues contain no displayable particles")
            positions = np.concatenate(all_positions)
            meta = {"dataset_id": identifier, "fingerprint": digest, "layers": layers,
                    "center_kpc": [0., 0., 0.], "sample_median_kpc": np.median(positions, axis=0).tolist(),
                    "provenance": provenance, "units": "kpc", "stride_floats": 8,
                    "sampling": "Deterministic uniform display sample. Stellar kernel radii query the full source layer; all sources and mesh sites remain in radiative transfer.",
                    "coloring": "Stars use age colour and a mass/age light proxy; young sources use SFR; dust uses dust mass. The projected cubic-spline display is structural, not band photometry or dust transfer."}
            (cache_dir / "cloud.json").write_text(json.dumps(meta, indent=2))
            return cache_dir, meta

    def raw_cloud(self,dataset):
        digest,provenance=dataset.fingerprint();cache_dir=self.cache/f'{dataset.id}-{digest[:16]}-{self.max_points}'
        with self.lock:
            if (cache_dir/'cloud.json').is_file():return cache_dir,json.loads((cache_dir/'cloud.json').read_text())
            cache_dir.mkdir(parents=True,exist_ok=True);layers=[];positions=[]
            with h5py.File(dataset.path) as raw:
                hdr=raw['Header'].attrs;a,h,box=float(hdr['Time']),float(hdr['HubbleParam']),float(hdr['BoxSize']);scale=a/h
                cosmology=LambdaCDM(H0=100*h,Om0=float(hdr.get('Omega0',.3)),Ode0=float(hdr.get('OmegaLambda',.7)),
                                    Ob0=float(hdr.get('OmegaBaryon',.0486)),Tcmb0=2.73)
                center=np.asarray(dataset.entry['center_native']);half=dataset.entry['preview_fov_kpc']/2
                specs=[('stars','PartType4','GFM_InitialMass',None),('gas','PartType0','Masses',np.array([.72,.30,1.])),('dust','PartType0','GFM_Metallicity',np.array([.46,.34,.22]))]
                for index,(kind,group_name,weight_field,color) in enumerate(specs):
                    if group_name not in raw or weight_field not in raw[group_name]:continue
                    group=raw[group_name];rng=np.random.default_rng(12345+index);kept=np.empty((0,5));keys=np.empty(0);count=0;total=0.;lo=np.full(3,np.inf);hi=np.full(3,-np.inf);reference_positions=[]
                    for start in range(0,len(group['Coordinates']),250000):
                        xyz=(np.asarray(group['Coordinates'][start:start+250000],float)-center+box/2)%box-box/2;xyz*=scale
                        select=np.all(np.abs(xyz)<half,axis=1)
                        if kind=='stars':
                            formation=np.asarray(group['GFM_StellarFormationTime'][start:start+250000],dtype=float)
                            select&=(formation>0)&(formation<=a)
                        if not select.any():continue
                        weight=np.asarray(group[weight_field][start:start+250000],float)[select]
                        if kind=='dust':weight*=np.asarray(group['Masses'][start:start+250000],float)[select]
                        extra=(formation[select] if kind=='stars' else
                               np.asarray(group['SubfindHsml'][start:start+250000],float)[select]*scale if 'SubfindHsml' in group else
                               np.zeros(select.sum()))
                        valid=np.isfinite(weight)&(weight>0)&np.isfinite(extra)&(extra>=0);values=np.column_stack((xyz[select][valid],weight[valid],extra[valid]))
                        if not len(values):continue
                        if kind=='stars':reference_positions.append(values[:,:3])
                        count+=len(values);total+=float(values[:,3].sum());lo=np.minimum(lo,values[:,:3].min(0));hi=np.maximum(hi,values[:,:3].max(0))
                        candidate=np.concatenate((kept,values));ranks=np.concatenate((keys,rng.random(len(values))));take=np.argpartition(ranks,self.max_points-1)[:self.max_points] if len(ranks)>self.max_points else np.arange(len(ranks));kept,keys=candidate[take],ranks[take]
                    if not count:continue
                    if kind=='stars':
                        age=(cosmology.lookback_time(1/kept[:,4]-1)-cosmology.lookback_time(1/a-1)).to_value('yr')
                        colors=stellar_colors(age);strength=stellar_strength(kept[:,3],age)
                        radii=neighbour_kernel_radii(kept[:,:3],np.concatenate(reference_positions))
                        kernel={"profile":"SKIRT projected cubic spline","neighbours":KERNEL_NEIGHBOURS,
                                "radius":"distance to 32nd nearest other source in full preview region"}
                    else:
                        colors=np.tile(color,(len(kept),1));strength=display_strength(kept[:,3]);radii=np.maximum(kept[:,4],0)
                        kernel={"profile":"soft projected proxy","radius":"SubfindHsml" if np.any(radii) else "screen-space fallback"}
                    name=f'{index}-{kind}.bin';write_cloud_layer(cache_dir/name,kept[:,:3],colors,radii,strength);positions.append(kept[:,:3])
                    layers.append(dict(kind=kind,count=count,positive_count=count,display_count=len(kept),file=name,bounds_kpc=[lo.tolist(),hi.tolist()],weight_total=total,weight_unit='raw code units',kernel=kernel,kernel_h_median_kpc=float(np.median(radii)) if np.any(radii) else None))
            if not layers:raise ValueError('The raw snapshot contains no displayable particles in the preview region')
            meta=dict(dataset_id=dataset.id,fingerprint=digest,layers=layers,center_kpc=[0.,0.,0.],sample_median_kpc=np.median(np.concatenate(positions),axis=0).tolist(),provenance=provenance,units='kpc',stride_floats=8,
                sampling='Raw HDF5 particles inside the preview cube; deterministic bounded display sample. Stellar radii query all valid stars in the region.',coloring='Stars use exact snapshot-cosmology ages plus a mass/age light proxy. Gas is a bright purple mass-weighted layer; dust remains a display-only metal-mass proxy.')
            (cache_dir/'cloud.json').write_text(json.dumps(meta,indent=2));return cache_dir,meta
