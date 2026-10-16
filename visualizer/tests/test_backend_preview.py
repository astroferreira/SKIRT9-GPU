import json
import io
import os
import shutil
import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from PIL import Image

from observatory.backend_preview import BackendFrame, BackendControls
from observatory.gpu_renderer import GPURenderer


def view(name='main', size=24):
    projection=np.eye(4);projection[2,2]=-2;projection[2,3]=-3
    world=np.eye(4);world[2,3]=3
    return dict(name=name,width=size,height=size,inverse_projection=projection.T.ravel().tolist(),
                camera_world=world.T.ravel().tolist())


def payload(**kwargs):
    return dict(session_id='a'*32,volume_key='b'*64,sequence=1,views=[view()],
                controls=BackendControls().model_dump())|kwargs


def fixture_volume(root):
    folder=root/('b'*64);folder.mkdir(parents=True)
    n=8
    stars=np.ones((n,n,n,4),dtype='<f4');stars[...,:3]=3
    stars.tofile(folder/'stars-dust.bin')
    np.zeros_like(stars).tofile(folder/'young.bin')
    (folder/'volume.json').write_text(json.dumps(dict(size=n,bounds_kpc=[[-1,-1,-1],[1,1,1]],
                                                   extinction_per_kpc=[.1,.2,.4])))


def test_backend_request_bounds():
    assert BackendFrame(**payload()).views[0].width == 24
    assert BackendControls(exposure=2e-7).exposure==2e-7
    for bad in [payload(volume_key='../escape'),payload(views=[view(),view()]),
                payload(views=[view(size=2049)]),payload(views=[view(size=2048),view('mini',2048)]),
                payload(controls={'dust_scale':float('nan')}),payload(controls={'steps':10000})]:
        with pytest.raises(ValidationError):
            BackendFrame(**bad)


def test_backend_session_security_and_limits(tmp_path,monkeypatch):
    from observatory.app import create_app
    config=tmp_path/'datasets.json';config.write_text('{"datasets":[]}')
    app=create_app(config=config,storage=tmp_path/'storage',allowed_hosts=['*'])
    backend=app.state.backend_preview
    monkeypatch.setattr(backend,'call',lambda *args,**kwargs:{'renderer':'test-only'})
    assert (backend.key_path.stat().st_mode & 0o777) == 0o600
    with TestClient(app,base_url='http://remote.example') as client:
        assert client.get('/api/backend-preview/status').json()['auth_required'] is True
        assert client.post('/api/backend-preview/sessions').status_code == 401
        headers={'Authorization':f'Bearer {backend.access_key}'}
        assert client.post('/api/backend-preview/sessions',headers=headers|{'Origin':'http://evil.example'}).status_code==403
        response=client.post('/api/backend-preview/sessions',headers=headers)
        assert response.status_code==200 and response.headers['cache-control']=='no-store'
        session=response.json();identifier=session['session_id']
        assert client.post('/api/backend-preview/frame',json=payload(session_id=identifier)).status_code==401
        token={'Authorization':f'Bearer {session["token"]}'}
        backend.sessions[identifier]['sequence']=5
        assert client.post('/api/backend-preview/frame',json=payload(session_id=identifier),headers=token).status_code==409
        assert client.delete(f'/api/backend-preview/sessions/{identifier}',headers=token).status_code==200
        assert identifier not in backend.sessions
        for _ in range(4):
            assert client.post('/api/backend-preview/sessions',headers=headers).status_code==200
        assert client.post('/api/backend-preview/sessions',headers=headers).status_code==429
        for session in backend.sessions.values():session['last']=time.monotonic()-121
        assert client.post('/api/backend-preview/sessions',headers=headers).status_code==200


def test_loopback_access_requires_loopback_host(tmp_path,monkeypatch):
    from observatory.app import create_app
    config=tmp_path/'datasets.json';config.write_text('{"datasets":[]}')
    app=create_app(config=config,storage=tmp_path/'storage',allowed_hosts=['*'])
    monkeypatch.setattr(app.state.backend_preview,'call',lambda *args,**kwargs:{})
    with TestClient(app,base_url='http://localhost',client=('127.0.0.1',54321)) as client:
        assert client.post('/api/backend-preview/sessions').status_code==200
        assert client.post('/api/backend-preview/sessions',headers={'Host':'external.example'}).status_code==401


@pytest.fixture
def gpu(tmp_path):
    if os.environ.get('SKIRT_BACKEND_GPU_TESTS')!='1':
        pytest.skip('Opt in to actual NVIDIA EGL tests with SKIRT_BACKEND_GPU_TESTS=1')
    fixture_volume(tmp_path)
    renderer=GPURenderer(tmp_path,device=int(os.environ.get('SKIRT_VIEWER_PREVIEW_GPU','1')))
    yield renderer
    renderer.close()


def test_nvidia_linear_transfer_and_dust_controls(gpu):
    request=BackendFrame(**payload()).model_dump()
    linear=gpu.render_view('test',request,request['views'][0],linear=True)
    opacity=np.array([.1,.2,.4])
    expected=3*(-np.expm1(-2*opacity))/opacity
    np.testing.assert_allclose(linear[12,12],expected,rtol=2e-5)
    for _ in range(7):linear=gpu.render_view('test',request,request['views'][0],linear=True)
    np.testing.assert_allclose(linear[12,12],expected,rtol=2e-5)
    request['controls']['dust']=False
    np.testing.assert_allclose(gpu.render_view('test',request,request['views'][0],linear=True)[12,12],[6,6,6],rtol=2e-5)
    request['controls']['stars']=False
    np.testing.assert_allclose(gpu.render_view('test',request,request['views'][0],linear=True),0,atol=1e-7)
    assert 'NVIDIA' in gpu.info['vendor']


def test_gpu_packet_shared_volume_cache_and_cleanup(gpu):
    request=BackendFrame(**payload(views=[view(),view('mini')])).model_dump()
    packet=gpu.render('test',request)
    length=struct.unpack('<I',packet[:4])[0]
    meta=json.loads(packet[4:4+length])
    assert meta['gpu']['vendor']=='NVIDIA Corporation'
    assert meta['resident_bytes']==8**3*32
    assert len(gpu.volumes)==1 and len(gpu.states)==2
    assert packet[4+length:4+length+8]==b'\x89PNG\r\n\x1a\n'
    assert sum(v['bytes'] for v in meta['views'])+4+length==len(packet)
    gpu.release_session('test');assert not gpu.states
    gpu.ttl=-1;gpu.evict();assert not gpu.volumes


def test_gpu_texture_budget_guard(gpu):
    gpu.budget=1
    with pytest.raises(ValueError,match='texture budget'):
        gpu.volume('b'*64)
    assert not gpu.volumes


def test_gpu_mixed_view_sizes_and_released_framebuffers(gpu):
    request=BackendFrame(**payload(views=[view(size=24),view('mini',12)])).model_dump()
    for _ in range(8):packet=gpu.render('test',request)
    length=struct.unpack('<I',packet[:4])[0];meta=json.loads(packet[4:4+length]);offset=4+length
    centers=[]
    for item in meta['views']:
        pixels=np.asarray(Image.open(io.BytesIO(packet[offset:offset+item['bytes']])));offset+=item['bytes']
        centers.append(pixels[pixels.shape[0]//2,pixels.shape[1]//2])
    np.testing.assert_allclose(centers[0],centers[1],atol=1)
    for item in request['views']:
        assert gpu.states[('test',item['name'])]['display'].viewport==(0,0,item['width'],item['height'])
    assert gpu.ctx.error=='GL_NO_ERROR'
    gpu.release_session('test')
    assert gpu.ctx.error=='GL_NO_ERROR'
    shutil.copytree(gpu.root/('b'*64),gpu.root/('c'*64))
    gpu.budget=8**3*32  # Force eviction of the first texture pair.
    request['volume_key']='c'*64
    gpu.render('new-session',request)
    assert gpu.ctx.error=='GL_NO_ERROR'
    assert len(gpu.volumes)==1


def test_worker_process_and_frame_api(tmp_path):
    if os.environ.get('SKIRT_BACKEND_GPU_TESTS')!='1':
        pytest.skip('Opt in to actual NVIDIA worker-process tests')
    from observatory.app import create_app
    config=tmp_path/'datasets.json';config.write_text('{"datasets":[]}')
    storage=tmp_path/'storage';fixture_volume(storage/'catalogues/browser-volumes')
    app=create_app(config=config,storage=storage)
    with TestClient(app,base_url='http://localhost',client=('127.0.0.1',54321)) as client:
        session=client.post('/api/backend-preview/sessions').json()
        response=client.post('/api/backend-preview/frame',json=payload(session_id=session['session_id']),
                             headers={'Authorization':f'Bearer {session["token"]}'})
        assert response.status_code==200,response.text
        assert response.headers['content-type']=='application/octet-stream'
        assert app.state.backend_preview.process.is_alive()
    assert app.state.backend_preview.process is None
