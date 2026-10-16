"""Runs in the exporter environment to materialize one cached physical region."""
from pathlib import Path
import argparse,json,subprocess,sys
parser=argparse.ArgumentParser();parser.add_argument('request_json');args=parser.parse_args();request=json.loads(Path(args.request_json).read_text())
exporter=Path(request['exporter']);sys.path.insert(0,str(exporter))
from physical_model import export_snapshot
options=request.get('physical_options',dict(diffuse_dust='torrey',diffuse_grain_mix='dl07'))
prefix=Path(request['prefix']);report=export_snapshot(request['snapshot'],request['center_native'],request['model_fov_kpc'],prefix,
    dust=options['diffuse_dust'],grain_mix=options['diffuse_grain_mix'])
report.update(raw_viewer_export=True,galaxy_center_native=request['galaxy_center_native'],camera_target_kpc=request['camera_target_kpc'],selection='Physical cube centered on camera target',snapshot_number=request['snapshot_number'])
Path(str(prefix)+'_physics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
command=[sys.executable,str(exporter/'skirt9_from_dat.py'),'--prefix',str(prefix),'--output-dir',str(prefix.parent/'skirt'),'--physics-model','bpass-toddlers','--instrument-mode','both','--num-packets','10000000','--fov-pc',str(request['model_fov_kpc']*1000),'--pixel-size-pc',str(request['model_fov_kpc']*1000/600),'--ski-label','physical_v1','--redshift',str(report['snapshot_redshift']),'--reduced-hubble-constant',str(report['hubble_param']),'--matter-density-fraction',str(report['omega_m'])]
command.extend(['--diffuse-dust',options['diffuse_dust'],'--diffuse-grain-mix',options['diffuse_grain_mix']])
subprocess.run(command,cwd=exporter,check=True)
(prefix.parent/'complete.json').write_text(json.dumps(dict(report=report,base_ski=str(prefix.parent/'skirt'/f'{prefix.name}_physical_v1.ski')),indent=2,allow_nan=False))
