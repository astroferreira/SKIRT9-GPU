"""Render an existing cached volume through the real backend API; never build or run SKIRT."""
import argparse
import io
import json
import struct
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('key')
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    meta=json.loads((root/'var/catalogues/browser-volumes'/args.key/'volume.json').read_text())
    bounds=np.asarray(meta['bounds_kpc']);center=bounds.mean(axis=0);span=float(np.max(bounds[1]-bounds[0]))
    projection=np.diag([span/2,span/2,-2*span,1.]);projection[2,3]=-2*span
    world=np.eye(4);world[:3,3]=center+[0,0,2*span]
    views=[dict(name=name,width=size,height=size,inverse_projection=projection.T.ravel().tolist(),
                camera_world=world.T.ravel().tolist()) for name,size in [('main',512),('mini',128)]]
    parent=root/'var/backend-validation';parent.mkdir(exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='render-',dir=parent))
    with httpx.Client(base_url=args.url,timeout=120) as client:
        before=client.get('/api/jobs').json()['jobs']
        response=client.post('/api/backend-preview/sessions');response.raise_for_status();session=response.json()
        headers={'Authorization':f'Bearer {session["token"]}'}
        results=[]
        try:
            for sequence in range(1,9):
                start=time.perf_counter()
                response=client.post('/api/backend-preview/frame',headers=headers,json=dict(session_id=session['session_id'],
                    volume_key=args.key,sequence=sequence,views=views,controls={'steps':192}))
                if response.is_error:
                    raise RuntimeError(f'Backend frame {response.status_code}: {response.text}')
                data=response.content
                elapsed=(time.perf_counter()-start)*1000
                length=struct.unpack('<I',data[:4])[0];result=json.loads(data[4:4+length])
                result.update(round_trip_ms=elapsed,transfer_bytes=len(data));results.append(result)
                if sequence==8:
                    offset=4+length
                    for view in result['views']:
                        png=data[offset:offset+view['bytes']];offset+=view['bytes']
                        (destination/f'{view["name"]}.png').write_bytes(png)
                        pixels=np.asarray(Image.open(io.BytesIO(png)))
                        assert pixels.max()>10 and pixels.mean()>0,'Blank backend image'
                        view['pixel_mean']=float(pixels.mean());view['pixel_max']=int(pixels.max())
                    assert offset==len(data)
                    means=[item['pixel_mean'] for item in result['views']]
                    assert .5 < means[1]/means[0] < 2.,'Main/mini brightness differs unexpectedly for the same camera'
        finally:
            client.delete(f'/api/backend-preview/sessions/{session["session_id"]}',headers=headers)
        after=client.get('/api/jobs').json()['jobs']
        assert [job['id'] for job in before]==[job['id'] for job in after]
    report=dict(volume_key=args.key,dataset_id=meta['dataset_id'],grid_size=meta['size'],bounds_kpc=meta['bounds_kpc'],
                frames=results,skirt_jobs_unchanged=True,notes='NVIDIA render plus PNG encoding and localhost transport; not GPU-only timing or physical SKIRT parity.')
    (destination/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(output=str(destination),grid_size=meta['size'],gpu=results[-1]['gpu'],
                          first_load_ms=results[0]['load_ms'],last_round_trip_ms=results[-1]['round_trip_ms'],
                          last_render_encode_ms=results[-1]['render_encode_ms'],resident_mib=results[-1]['resident_bytes']/1024**2,
                          samples=[v['samples'] for v in results[-1]['views']]),indent=2))


if __name__=='__main__':
    main()
