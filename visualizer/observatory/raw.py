"""Cache physical exports derived from raw AREPO snapshots on observation demand."""
from pathlib import Path
import hashlib,json,subprocess,sys,uuid
import numpy as np
from .catalog import Dataset
from .physics import physical_options

EXPORTER=Path('/home/astroferrreira/src/illustris_to_stellarmap')
EXPORTER_PYTHON=EXPORTER/'.venv/bin/python'

def materialize(dataset,request,cache):
    cache=Path(cache).resolve()
    model_fov=request.model_fov_kpc or dataset.entry['default_model_fov_kpc']
    if {'image', 'rgb'} & set(request.selected_instruments()) and model_fov < request.camera.fov_kpc:
        raise ValueError('Physical model region must be at least as large as the camera FOV')
    base=np.asarray(dataset.entry['center_native'],float);scale=dataset.entry['scale_factor']/dataset.entry['hubble_param'];box=dataset.entry['box_size_native']
    center=(base+np.asarray(request.camera.target_kpc)/scale)%box
    options = physical_options(dataset.entry)
    fingerprint,_=dataset.fingerprint();payload=dict(fingerprint=fingerprint,center_native=center.tolist(),model_fov_kpc=model_fov,physical_options=options,cache_format=3)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();target=cache/'raw-exports'/f'{dataset.id}-{digest[:20]}'
    complete=target/'complete.json'
    reused=complete.is_file()
    if not reused:
        temporary=target.parent/f'.{target.name}-{uuid.uuid4().hex}.partial';temporary.mkdir(parents=True)
        prefix=temporary/dataset.id;spec=dict(exporter=str(EXPORTER),snapshot=str(dataset.path),prefix=str(prefix),center_native=center.tolist(),galaxy_center_native=base.tolist(),camera_target_kpc=list(request.camera.target_kpc),model_fov_kpc=model_fov,snapshot_number=dataset.entry['snapshot'])
        spec['physical_options'] = options
        spec_path=temporary/'request.json';spec_path.write_text(json.dumps(spec,indent=2))
        with (temporary/'export.log').open('w') as log:subprocess.run([str(EXPORTER_PYTHON),str(Path(__file__).with_name('raw_worker.py')),str(spec_path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        if target.exists():raise RuntimeError('On-demand export cache target appeared concurrently')
        temporary.rename(target)
        # The generator records absolute catalogue paths. Retarget them after the
        # atomic partial-directory rename so the cached model remains self-contained.
        for ski in (target/'skirt').glob('*.ski'):
            ski.write_text(ski.read_text().replace(str(temporary),str(target)))
        complete.write_text(complete.read_text().replace(str(temporary),str(target)))
    record=json.loads(complete.read_text());base_ski=target/'skirt'/f'{dataset.id}_physical_v1.ski';entry=dict(dataset.entry,id=dataset.id,name=dataset.entry['name'],base_ski=str(base_ski));entry.pop('raw_snapshot',None)
    derived=Dataset(entry,Path('/'))
    run_request=request.model_copy(update={'camera':request.camera.model_copy(update={'target_kpc':(0.,0.,0.)})})
    return derived,run_request,dict(cache_key=digest,cache_directory=str(target),model_fov_kpc=model_fov,center_native=center.tolist(),reused=reused,physical_report=record['report'])
