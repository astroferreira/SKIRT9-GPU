"""Immutable, opt-in source-velocity revisions of existing physical catalogues."""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import h5py

from .catalog import Dataset, RawDataset, stamp
from .raw import EXPORTER, EXPORTER_PYTHON


@lru_cache(maxsize=512)
def _missing_velocity_fields(path, size, mtime_ns, gas_needed):
    # Cache only HDF5 metadata inspection, keyed by the raw file identity.
    with h5py.File(path) as snapshot:
        for group in ('PartType4', 'PartType0') if gas_needed else ('PartType4',):
            field = f'{group}/Velocities'
            if field not in snapshot:
                return f'Missing snapshot field {field}; download velocities for this cutout first'
            if snapshot[field].ndim != 2 or snapshot[field].shape[1] != 3:
                return f'Invalid snapshot velocity shape: {field}'
    return None


def availability(dataset):
    try:
        if isinstance(dataset, RawDataset):
            raw, gas_needed = dataset.path, True
        else:
            metadata = dataset.entry.get('physics', {}).get('metadata_path')
            if not metadata:
                raise ValueError('Source velocities require physical export provenance and a raw snapshot')
            metadata = Path(metadata)
            report = json.loads(metadata.read_text())
            fields = metadata.with_name(metadata.name.removesuffix('_physics.json')+'_physical_fields.npz')
            if not fields.is_file():
                raise ValueError('Missing saved physical particle selections for source velocities')
            raw, gas_needed = Path(report['raw_path']), report.get('toddlers_count', 1) > 0
        identity = stamp(raw)
        missing = _missing_velocity_fields(str(raw), identity['bytes'], identity['mtime_ns'], gas_needed)
        if missing:
            raise ValueError(missing)
        return dict(available=True)
    except (OSError, ValueError, KeyError) as error:
        return dict(available=False, reason=str(error))


def input_identity(dataset, metadata=None):
    identity = {'scripts': {name: hashlib.sha256((EXPORTER/name).read_bytes()).hexdigest()
                           for name in ('source_kinematics.py', 'physical_model.py')}}
    if isinstance(dataset, RawDataset):
        identity['raw'] = stamp(dataset.path)
        return identity
    metadata = metadata or dataset.entry.get('physics', {}).get('metadata_path')
    if not metadata:
        raise ValueError('Source velocities require physical export provenance and a raw snapshot')
    metadata = Path(metadata)
    report = json.loads(metadata.read_text())
    fields = metadata.with_name(metadata.name.removesuffix('_physics.json')+'_physical_fields.npz')
    identity.update(metadata=str(metadata), metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    raw=stamp(Path(report['raw_path'])), physical_fields=stamp(fields))
    return identity


def materialize(dataset, cache, metadata=None):
    identity = input_identity(dataset, metadata)
    fingerprint, _ = dataset.fingerprint()
    digest = hashlib.sha256(json.dumps({'model': fingerprint, 'kinematics': identity,
                                       'format': 1}, sort_keys=True).encode()).hexdigest()
    parent = Path(cache)/'source-velocities'
    target = parent/f'{dataset.id}-{digest[:20]}'
    reused = (target/'kinematics.json').is_file()
    if not reused:
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.velocity-', dir=parent) as temporary:
            output = Path(temporary)/'model'
            log_path = Path(temporary)/'export.log'
            with log_path.open('w') as log:
                run = subprocess.run([str(EXPORTER_PYTHON), str(EXPORTER/'source_kinematics.py'),
                    str(dataset.path), identity['metadata'], str(output)], stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                raise ValueError('Source velocity export failed: '+log_path.read_text()[-3000:])
            log_path.rename(output/'export.log')
            output.rename(target)
    report = json.loads((target/'kinematics.json').read_text())
    entry = copy.deepcopy(dataset.entry)
    entry.update(base_ski=str(target/'model.ski'))
    entry.setdefault('physics', {})['kinematics'] = report
    return Dataset(entry, Path('/')), dict(report=report, cache_key=digest,
                                         cache_directory=str(target), reused=reused)
