import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import h5py
from astropy.io import fits
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from observatory.app import create_app
from observatory.catalog import Catalog, RawDataset, neighbour_kernel_radii
from observatory.raw import materialize
from observatory.jobs import Jobs, execution_evidence, execution_limit, progress
from observatory.simulation import QUALITY, Camera, RenderRequest, camera_basis, prepare, render_fits, render_rgb_fits


@pytest.fixture
def fixture(tmp_path):
    # Deliberately asymmetric positions catch clipping, first-N sampling and flips.
    values = np.array([[i*100, -i*250, i*75, 2, 10, .02, 1e9] for i in range(101)])
    np.savetxt(tmp_path / "stars.dat", values)
    np.savetxt(tmp_path / "dust.dat", [[100,200,300,9,1000],[200,400,600,8,2000]])
    (tmp_path / "base.ski").write_text('''<skirt-simulation-hierarchy><MonteCarloSimulation simulationMode="DustEmission" numPackets="1000000000"><random><Random seed="1"/></random><cosmology><FlatUniverseCosmology redshift="1"/></cosmology><SourceSystem minWavelength="0.01 micron" maxWavelength="30 micron"><ParticleSource filename="stars.dat"><BruzualCharlotSEDFamily resolution="High"/></ParticleSource></SourceSystem><MediumSystem><VoronoiMeshMedium filename="dust.dat"/><DustEmissionOptions dustEmissionType="Stochastic"/></MediumSystem><InstrumentSystem><defaultWavelengthGrid/><instruments><FrameInstrument instrumentName="old" distance="0 Mpc" fieldOfViewX="60000 pc"/><SEDInstrument/></instruments></InstrumentSystem></MonteCarloSimulation></skirt-simulation-hierarchy>''')
    (tmp_path / "datasets.json").write_text(json.dumps({"datasets":[{"id":"test", "name":"Test galaxy", "base_ski":"base.ski", "legacy_dust_columns":[0,1,2,4]}]}))
    catalog = Catalog(tmp_path / "datasets.json", tmp_path / "cache", max_points=20)
    return tmp_path, catalog


def test_cloud_uniform_bounded_and_full_counts(fixture):
    root, catalog = fixture
    folder, meta = catalog.cloud("test")
    stars = meta["layers"][0]
    assert stars["count"] == 101 and stars["display_count"] == 20
    values = np.fromfile(folder / stars["file"], dtype="<f4").reshape(-1, 8)
    assert values[:, 0].max() > 7  # not merely the first 20 rows
    assert np.all(values[:, 6] > 0) and np.all(values[:, 7] > 0)
    assert stars['kernel']['neighbours'] == 32 and meta['stride_floats'] == 8
    assert stars["bounds_kpc"] == [[0., -25., 0.], [10., 0., 7.5]]
    assert catalog.cloud("test")[1] == meta
    with (root / "stars.dat").open("a") as stream:
        stream.write("10200 -25500 7650 2 10 .02 1e9\n")
    assert catalog.cloud("test")[1]["fingerprint"] != meta["fingerprint"]


def test_velocity_request_is_opt_in_and_has_distinct_cache_identity():
    request=RenderRequest(dataset_id='test')
    assert not request.source_velocities and 'source_velocities' not in request.render_identity()
    moving=request.model_copy(update={'source_velocities':True})
    assert moving.render_identity()['source_velocities'] is True
    assert moving.render_identity()!=request.render_identity()
    with pytest.raises(ValidationError):RenderRequest(dataset_id='test',source_velocities='true')


def test_velocity_source_columns_preserve_preview_physical_quantities(fixture):
    root,catalog=fixture
    original_folder,original=catalog.cloud('test')
    original_points=np.fromfile(original_folder/original['layers'][0]['file'],dtype='<f4').copy()
    stars=np.loadtxt(root/'stars.dat'); velocities=np.tile([-200.,30.,-100.],(len(stars),1))
    np.savetxt(root/'stars.dat',np.column_stack((stars[:,:4],velocities,stars[:,4:])))
    tree=ET.parse(root/'base.ski');tree.find('.//ParticleSource').set('importVelocity','true');tree.write(root/'base.ski')
    dataset=catalog.datasets['test'];table=dataset.tables()[0]
    assert table['weight_column']==7 and table['velocity_columns']==3
    folder,updated=catalog.cloud('test')
    np.testing.assert_array_equal(np.fromfile(folder/updated['layers'][0]['file'],dtype='<f4'),original_points)
    assert updated['fingerprint']!=original['fingerprint']


def test_velocity_request_without_provenance_rejected_before_queue(fixture):
    root,catalog=fixture
    jobs=Jobs(catalog,root/'jobs',Path(sys.executable))
    try:
        with pytest.raises(ValueError,match='provenance'):
            jobs.submit(RenderRequest(dataset_id='test',source_velocities=True))
        assert not jobs.list()
    finally:jobs.close()


def test_velocity_overlay_cache_and_relative_paths(fixture, monkeypatch):
    from types import SimpleNamespace
    from observatory import kinematics
    root,catalog=fixture
    exporter=root/'exporter';exporter.mkdir()
    for name in ('source_kinematics.py','physical_model.py'):(exporter/name).write_text('# test exporter\n')
    monkeypatch.setattr(kinematics,'EXPORTER',exporter)
    raw=root/'raw.hdf5';raw.write_bytes(b'raw')
    metadata=root/'model_physics.json';metadata.write_text(json.dumps({'raw_path':str(raw)}))
    (root/'model_physical_fields.npz').write_bytes(b'fields')
    dataset=catalog.datasets['test'];dataset.entry['physics']={'metadata_path':str(metadata)}
    calls=[]
    def export(command, **kwargs):
        calls.append(command)
        output=Path(command[-1]);output.mkdir()
        tree=dataset.tree()
        for node in tree.iter():
            if node.get('filename'):node.set('filename',str((root/node.get('filename')).resolve()))
        source=tree.find('.//ParticleSource');source.set('filename','stars_velocity.dat');source.set('importVelocity','true')
        values=np.loadtxt(root/'stars.dat')
        np.savetxt(output/'stars_velocity.dat',np.column_stack((values[:,:4],np.zeros((len(values),3)),values[:,4:])))
        tree.write(output/'model.ski')
        (output/'kinematics.json').write_text(json.dumps({'mode':'sources'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(kinematics.subprocess,'run',export)
    derived,first=kinematics.materialize(dataset,catalog.cache)
    again,second=kinematics.materialize(dataset,catalog.cache)
    assert len(calls)==1 and not first['reused'] and second['reused']
    assert first['cache_key']==second['cache_key'] and derived.info()['available']
    assert derived.tables()[0]['path'].is_file()
    assert 'kinematics' not in dataset.entry['physics']
    raw.write_bytes(b'changed raw')
    _,third=kinematics.materialize(dataset,catalog.cache)
    assert len(calls)==2 and third['cache_key']!=first['cache_key']


def test_missing_velocity_fields_rejected_before_queue_and_cache_refreshed(fixture):
    from observatory.kinematics import availability
    root,catalog=fixture
    raw=root/'raw.hdf5'
    with h5py.File(raw,'w') as handle:
        handle.create_group('PartType4')
    metadata=root/'model_physics.json'
    metadata.write_text(json.dumps({'raw_path':str(raw),'toddlers_count':0}))
    (root/'model_physical_fields.npz').write_bytes(b'fields')
    dataset=catalog.datasets['test'];dataset.entry['physics']={'metadata_path':str(metadata)}
    assert not availability(dataset)['available']
    jobs=Jobs(catalog,root/'jobs',Path(sys.executable))
    try:
        with pytest.raises(ValueError,match='PartType4/Velocities'):
            jobs.submit(RenderRequest(dataset_id='test',source_velocities=True))
        assert not jobs.list()
    finally:jobs.close()
    with h5py.File(raw,'a') as handle:
        handle['PartType4']['Velocities']=[[0.,0.,0.]]
    assert availability(dataset)['available']


def test_raw_snapshot_series_central_selection_periodicity_and_proxy(tmp_path):
    output=tmp_path/'output';output.mkdir()
    snapshot=output/'snap_007.hdf5';groups=output/'fof_subhalo_tab_007.hdf5'
    with h5py.File(snapshot,'w') as handle:
        header=handle.create_group('Header').attrs
        header.update(Redshift=1.,Time=.5,HubbleParam=.5,BoxSize=100.)
        gas=handle.create_group('PartType0')
        gas['Coordinates']=[[99.,50.,50.],[20.,20.,20.]];gas['Masses']=[2.,3.];gas['GFM_Metallicity']=[.1,.2]
        stars=handle.create_group('PartType4')
        stars['Coordinates']=[[1.,50.,50.],[98.,50.,50.]];stars['GFM_InitialMass']=[4.,5.];stars['GFM_StellarFormationTime']=[.3,-.1]
    with h5py.File(groups,'w') as handle:
        group=handle.create_group('Group');group['GroupFirstSub']=[1]
        subhalo=handle.create_group('Subhalo');subhalo['SubhaloPos']=[[10.,10.,10.],[0.,50.,50.]]
    registry={'raw_series':[{'id_prefix':'raw','name':'Raw','output_dir':str(output),'preview_fov_kpc':10,'default_model_fov_kpc':20}],'datasets':[]}
    (tmp_path/'datasets.json').write_text(json.dumps(registry))
    catalog=Catalog(tmp_path/'datasets.json',tmp_path/'cache',max_points=20)
    dataset=catalog.datasets['raw-007'];assert isinstance(dataset,RawDataset)
    assert dataset.entry['subhalo']==1 and dataset.entry['center_native']==[0.,50.,50.]
    old_fingerprint = dataset.fingerprint()[0]
    dataset.entry['physical_options'] = dict(diffuse_dust='rr2014', diffuse_grain_mix='wd01')
    assert dataset.fingerprint()[0] != old_fingerprint
    folder,meta=catalog.cloud('raw-007');assert [layer['kind'] for layer in meta['layers']]==['stars','gas','dust']
    assert [layer['count'] for layer in meta['layers']]==[1,1,1]
    star=np.fromfile(folder/meta['layers'][0]['file'],dtype='<f4').reshape(-1,8)
    assert star[0,0]==pytest.approx(1.)  # wrapped across the periodic boundary, with a/h=1
    gas=np.fromfile(folder/meta['layers'][1]['file'],dtype='<f4').reshape(-1,8)
    assert gas[0,3:6].tolist()==pytest.approx([.72,.30,1.])
    request=RenderRequest(dataset_id='raw-007',camera=Camera(fov_kpc=21),model_fov_kpc=20)
    with pytest.raises(ValueError,match='at least as large'):
        materialize(dataset,request,tmp_path/'cache')
    manager=Jobs(catalog,tmp_path/'jobs',fake_worker(tmp_path))
    try:
        with pytest.raises(ValueError,match='at least as large'):
            manager.submit(request)
    finally:manager.close()


@pytest.mark.parametrize('option', [dict(diffuse_dust='invalid'), dict(diffuse_grain_mix='invalid'), dict(unknown='wd01')])
def test_raw_dust_options_reject_unknowns(option):
    from observatory.physics import physical_options
    with pytest.raises(ValueError):
        physical_options(dict(physical_options=option))


def test_raw_materialization_propagates_dust_options_and_separates_cache(tmp_path, monkeypatch):
    import observatory.raw as raw
    from observatory.physics import physical_options
    snapshot = tmp_path/'snapshot.hdf5'; snapshot.touch()
    group = tmp_path/'groups.hdf5'; group.touch()
    entry = dict(id='test', name='Test', raw_snapshot=str(snapshot), group_catalog=str(group), snapshot=1,
                 center_native=[0,0,0], scale_factor=1., hubble_param=.7, box_size_native=1000.,
                 preview_fov_kpc=50, default_model_fov_kpc=50, physics={})
    dataset = RawDataset(entry)
    request = RenderRequest(dataset_id='test', camera=Camera(fov_kpc=50))
    calls = []
    def worker(command, **kwargs):
        spec = json.loads(Path(command[-1]).read_text()); calls.append(spec)
        prefix = Path(spec['prefix']); folder=prefix.parent/'skirt'; folder.mkdir()
        base = folder/'test_physical_v1.ski'
        base.write_text('<model/>')
        (prefix.parent/'complete.json').write_text(json.dumps(dict(report=spec['physical_options'],base_ski=str(base))))
    monkeypatch.setattr(raw.subprocess, 'run', worker)
    _, _, first = materialize(dataset, request, tmp_path/'cache')
    _, _, repeat = materialize(dataset, request, tmp_path/'cache')
    assert repeat['reused'] and len(calls) == 1
    entry['physical_options'] = dict(diffuse_dust='rr2014',diffuse_grain_mix='wd01')
    _, _, second = materialize(dataset, request, tmp_path/'cache')
    assert len(calls) == 2 and first['cache_key'] != second['cache_key']
    assert calls[-1]['physical_options'] == physical_options(entry)
    assert Path(second['cache_directory'], 'skirt', 'test_physical_v1.ski').is_file()


@pytest.mark.parametrize("angles", [(0,0,90),(90,-90,0),(90,0,0),(121,-63,47),(180,180,0)])
def test_projection_and_model_preservation(fixture, angles):
    root,catalog=fixture
    out=root / "output";out.mkdir()
    request=RenderRequest(dataset_id="test",camera=Camera(inclination=angles[0],azimuth=angles[1],roll=angles[2],target_kpc=(5,-3,2),fov_kpc=24))
    manifest=prepare(catalog.datasets['test'],request,out,catalog.cache)
    tree=ET.parse(out/'observation.ski')
    frames=tree.findall('.//FrameInstrument');assert len(frames)==1
    frame=frames[0];assert frame.attrib['fieldOfViewX']=='24000 pc'
    right,up,observer=camera_basis(request.camera)
    np.testing.assert_allclose(np.cross(right,up),observer,atol=1e-14)
    assert float(frame.attrib['centerX'].split()[0])==pytest.approx(np.dot(right,[5000,-3000,2000]))
    assert float(frame.attrib['centerY'].split()[0])==pytest.approx(np.dot(up,[5000,-3000,2000]))
    assert tree.find('.//SourceSystem').attrib['maxWavelength']=='30 micron'
    assert tree.find('.//DustEmissionOptions').attrib['dustEmissionType']=='Stochastic'
    assert tree.find('.//FlatUniverseCosmology').attrib['redshift']=='1'
    assert tree.find('.//SEDInstrument') is None
    assert tree.find('.//BroadBand').attrib['bandName']==request.band
    np.testing.assert_array_equal(np.loadtxt(tree.find('.//VoronoiMeshMedium').attrib['filename'])[:,3],[1000,2000])
    np.testing.assert_array_equal(np.loadtxt(root/'dust.dat')[:,3],[9,8])
    assert manifest['packets_per_phase']==1000000


def test_fits_orientation_and_preserved_values(tmp_path):
    data=np.zeros((1,256,256));data[0,0,7]=10;data[0,255,2]=5
    path=tmp_path/'image.fits';fits.writeto(path,data,header=fits.Header({'BUNIT':'MJy/sr'}))
    before=path.read_bytes();meta=render_fits(path,tmp_path/'image.png',256)
    png=np.asarray(Image.open(tmp_path/'image.png'))
    assert png[255,7]>0 and png[0,7]==0 and png[0,2]>0
    assert path.read_bytes()==before and meta['sum']==15 and meta['units']=='MJy/sr'
    fits.writeto(path,np.full((256,256),np.nan),overwrite=True)
    with pytest.raises(ValueError):render_fits(path,tmp_path/'bad.png',256)


def test_rgb_fits_channel_mapping_orientation_and_preserved_values(tmp_path):
    cube=np.zeros((3,256,256));cube[0,0,7]=1;cube[1,0,7]=2;cube[2,0,7]=3
    path=tmp_path/'rgb.fits';fits.writeto(path,cube,header=fits.Header({'BUNIT':'MJy/sr'}))
    before=path.read_bytes();meta=render_rgb_fits(path,tmp_path/'rgb.png',256,{
        'red':'JWST_NIRCAM_F444W','green':'JWST_NIRCAM_F200W','blue':'JWST_NIRCAM_F115W'})
    png=np.asarray(Image.open(tmp_path/'rgb.png'))
    assert png[255,7].tolist()==[255,255,255] and png[0,7].tolist()==[0,0,0]
    assert path.read_bytes()==before and meta['units']=='MJy/sr'
    assert meta['fits_plane_order']==['JWST_NIRCAM_F115W','JWST_NIRCAM_F200W','JWST_NIRCAM_F444W']
    assert meta['channels']['red']['fits_plane']==2 and meta['display']['scientific_fits_preserved']


def test_multiple_instruments_share_one_simulation(fixture):
    root,catalog=fixture
    out=root/'multi';out.mkdir()
    request=RenderRequest(dataset_id='test',instruments=('image','rgb','sed'),output='rgb',spectral_mode='miri_lrs')
    manifest=prepare(catalog.datasets['test'],request,out,catalog.cache)
    tree=ET.parse(out/'observation.ski');system=tree.find('.//InstrumentSystem')
    assert [node.get('instrumentName') for node in system.find('instruments')]==['view','rgb','spectrum']
    assert len(tree.findall('.//FrameInstrument'))==2 and len(tree.findall('.//SEDInstrument'))==1
    rgb=tree.find(".//FrameInstrument[@instrumentName='rgb']")
    assert [node.get('bandName') for node in rgb.findall('.//BroadBand')]==[
        'JWST_NIRCAM_F444W','JWST_NIRCAM_F200W','JWST_NIRCAM_F115W']
    assert tree.find(".//SEDInstrument[@instrumentName='spectrum']/wavelengthGrid/ResolutionWavelengthGrid") is not None
    assert manifest['instruments']==['image','rgb','sed']
    assert manifest['rgb_observation']['fits_plane_order']==[
        'JWST_NIRCAM_F115W','JWST_NIRCAM_F200W','JWST_NIRCAM_F444W']


@pytest.mark.parametrize(('quality','packets','limit'),[
    ('billion',1_000_000_000,21_600),('ten_billion',10_000_000_000,86_400)])
def test_large_packet_budgets_reach_skirt_and_extend_timeout(fixture,quality,packets,limit):
    root,catalog=fixture
    out=root/quality;out.mkdir()
    request=RenderRequest(dataset_id='test',quality=quality)
    manifest=prepare(catalog.datasets['test'],request,out,catalog.cache)
    assert ET.parse(out/'observation.ski').getroot().find('.//MonteCarloSimulation').get('numPackets')==str(packets)
    assert manifest['packets_per_phase']==packets
    assert QUALITY[quality]['packets']==packets
    assert execution_limit(7200,quality)==limit


def test_rgb_request_requires_distinct_known_bands():
    with pytest.raises(ValidationError,match='distinct'):
        RenderRequest(dataset_id='test',instruments=('rgb',),rgb_bands={'red':'SLOAN_SDSS_R','green':'SLOAN_SDSS_R','blue':'SLOAN_SDSS_G'})
    with pytest.raises(ValidationError,match='Unknown broadband'):
        RenderRequest(dataset_id='test',instruments=('rgb',),rgb_bands={'red':'fake','green':'SLOAN_SDSS_R','blue':'SLOAN_SDSS_G'})
    with pytest.raises(ValidationError,match='only once'):
        RenderRequest(dataset_id='test',instruments=('image','image'))


@pytest.mark.parametrize('change',[{'camera':{'fov_kpc':0}}, {'camera':{'azimuth':float('nan')}}, {'camera':{'target_kpc':[10001,0,0]}}, {'pixels':99999}, {'seed':-1}, {'quality':'expensive'}, {'shell':'touch injected'}])
def test_request_rejects_invalid_parameters(change):
    with pytest.raises(ValidationError):RenderRequest.model_validate({'dataset_id':'test',**change})


def test_full_halo_camera_and_view_only_submission_guard(fixture):
    root,catalog=fixture
    camera=Camera(fov_kpc=8000,target_kpc=(-3671,1400,1800))
    out=root/'full-halo';out.mkdir()
    prepare(catalog.datasets['test'],RenderRequest(dataset_id='test',camera=camera),out,catalog.cache)
    assert ET.parse(out/'observation.ski').find('.//FrameInstrument').get('fieldOfViewX')=='8000000 pc'
    with pytest.raises(ValidationError):Camera(fov_kpc=10001)
    catalog.datasets['test'].entry.update(render_available=False,render_unavailable_reason='Full halo exceeds memory budget')
    assert catalog.datasets['test'].info()['render_available'] is False
    # Viewing remains available while both image and SED launches are rejected.
    assert catalog.cloud('test')[1]['layers']
    manager=Jobs(catalog,root/'jobs',fake_worker(root))
    try:
        for output in ('image','sed'):
            with pytest.raises(ValueError,match='exceeds memory'):
                manager.submit(RenderRequest(dataset_id='test',output=output,camera=camera))
        assert manager.list()==[]
    finally:manager.close()


def fake_worker(root, delay=0):
    executable=root/'fake-renderer'
    executable.write_text(f'''#!{sys.executable}
import sys,time
from pathlib import Path
import numpy as np
from astropy.io import fits
print("TEST ONLY: fake subprocess for lifecycle tests",flush=True)
time.sleep({delay})
folder=Path(sys.argv[sys.argv.index('-o')+1])
fits.writeto(folder/'observation_view_total.fits',np.ones((1,256,256)))
''')
    executable.chmod(0o755)
    return executable


def fake_multi_worker(root):
    executable=root/'fake-multi-renderer'
    executable.write_text(f'''#!{sys.executable}
import math,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from astropy.io import fits
folder=Path(sys.argv[sys.argv.index('-o')+1]);tree=ET.parse(folder/'observation.ski')
if tree.find(".//FrameInstrument[@instrumentName='view']") is not None:
    fits.writeto(folder/'observation_view_total.fits',np.ones((1,256,256)))
if tree.find(".//FrameInstrument[@instrumentName='rgb']") is not None:
    fits.writeto(folder/'observation_rgb_total.fits',np.stack([np.ones((256,256))*x for x in (1,2,3)]))
sed=tree.find(".//SEDInstrument[@instrumentName='spectrum']")
if sed is not None:
    grid=sed.find('wavelengthGrid/ResolutionWavelengthGrid');low=float(grid.get('minWavelength').split()[0]);high=float(grid.get('maxWavelength').split()[0]);r=float(grid.get('resolution'))
    n=1+math.ceil(math.log(high/low)/math.log1p(1/r));np.savetxt(folder/'observation_spectrum_sed.dat',np.column_stack((np.geomspace(low,high,n),np.ones(n)*1e-15)),header='column 1: wavelength; lambda (micron)\\ncolumn 2: total flux; lambda*F_lambda (W/m2)')
print('TEST ONLY: fake multi-instrument subprocess')
''')
    executable.chmod(0o755)
    return executable


def wait_for(manager, identifier, states, timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        job=manager.get(identifier)
        if job['status'] in states:return job
        time.sleep(.05)
    pytest.fail(f"Job never reached {states}: {manager.get(identifier)}")


def test_worker_completion_cache_and_restart(fixture):
    root,catalog=fixture
    executable=fake_worker(root)
    jobs=Jobs(catalog,root/'jobs',executable)
    request=RenderRequest(dataset_id='test',pixels=256)
    try:
        first,_=jobs.submit(request)
        complete=wait_for(jobs,first['id'],{'completed','failed'})
        assert complete['status']=='completed',complete
        assert complete['gpu_confirmed'] is False
        second,reused=jobs.submit(request)
        assert reused and first['id']==second['id']
    finally:jobs.close()
    again=Jobs(catalog,root/'jobs',executable)
    try:assert again.get(first['id'])['status']=='completed'
    finally:again.close()


def test_multi_instrument_job_products_and_downloads(fixture):
    root,catalog=fixture
    app=create_app(root/'datasets.json',root/'multi-state',fake_multi_worker(root))
    request={'dataset_id':'test','pixels':256,'instruments':['image','rgb','sed'],'output':'rgb',
             'spectral_mode':'miri_lrs','rgb_bands':{'red':'JWST_NIRCAM_F444W','green':'JWST_NIRCAM_F200W','blue':'JWST_NIRCAM_F115W'}}
    with TestClient(app) as client:
        response=client.post('/api/jobs',json=request);assert response.status_code==202,response.text
        identifier=response.json()['job']['id'];job=wait_for(app.state.jobs,identifier,{'completed','failed'})
        assert job['status']=='completed',job
        assert set(job['products'])=={'image','rgb','sed'} and job['primary_product']=='rgb'
        assert job['rgb']['fits_plane_order']==['JWST_NIRCAM_F115W','JWST_NIRCAM_F200W','JWST_NIRCAM_F444W']
        for product in job['products'].values():
            assert client.get(product['preview_url']).status_code==200
            assert client.get(product['fits_url']).status_code==200
        assert client.get(job['products']['sed']['spectrum_url']).status_code==200


def test_worker_gpu_feeding_and_bounded_dust_configuration(fixture, monkeypatch):
    root, catalog = fixture
    monkeypatch.setenv("SKIRTGPU_SOURCE_LAUNCH", "1")
    monkeypatch.setenv("SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS", "999999")
    monkeypatch.setenv("SKIRTGPU_PROFILE_BATCH", "1")
    jobs = Jobs(catalog, root / 'jobs', fake_worker(root), threads=16, dust_batch_cells=512)
    try:
        job, _ = jobs.submit(RenderRequest(dataset_id='test', pixels=256))
        complete = wait_for(jobs, job['id'], {'completed', 'failed'})
        assert complete['status'] == 'completed', complete
        manifest = json.loads((root / 'jobs' / job['id'] / 'manifest.json').read_text())
        assert manifest['environment'] == {
            'SKIRTGPU_PRECOMPUTE_SOURCE_SPECTRA': '0',
            'SKIRTGPU_BATCH_FIRST_FORCED_CHUNK': '2097152',
            'SKIRTGPU_PRIMARY_LIFECYCLE_THREADS': '16',
            'SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS': '512',
        }
    finally:
        jobs.close()


@pytest.mark.parametrize('batch', [0, -1, 1048577, True, 256.5, '512'])
def test_worker_rejects_invalid_dust_batch_size(fixture, batch):
    root, catalog = fixture
    with pytest.raises(ValueError, match='between 1 and 1048576'):
        Jobs(catalog, root / 'jobs', root / 'unused', dust_batch_cells=batch)
    assert not (root / 'jobs').exists()


def test_running_and_queued_cancellation(fixture):
    root,catalog=fixture
    jobs=Jobs(catalog,root/'jobs',fake_worker(root,delay=15))
    try:
        first,_=jobs.submit(RenderRequest(dataset_id='test',pixels=256))
        wait_for(jobs,first['id'],{'running'})
        second,_=jobs.submit(RenderRequest(dataset_id='test',pixels=256,seed=2))
        jobs.cancel(second['id']);assert jobs.get(second['id'])['status']=='cancelled'
        jobs.cancel(first['id']);assert wait_for(jobs,first['id'],{'cancelled'})['cancel_requested']
        assert not (root/'jobs'/second['id']/'observation.ski').exists()
    finally:jobs.close()


def test_timeout_terminates_subprocess(fixture):
    root,catalog=fixture
    jobs=Jobs(catalog,root/'jobs',fake_worker(root,delay=15),timeout=.2)
    try:
        item,_=jobs.submit(RenderRequest(dataset_id='test',pixels=256))
        failed=wait_for(jobs,item['id'],{'failed'})
        assert 'execution limit' in failed['error']
    finally:jobs.close()


def test_api_boundaries_and_metadata(fixture):
    root,catalog=fixture
    app=create_app(root/'datasets.json',root/'state',fake_worker(root))
    with TestClient(app) as client:
        config=client.get('/api/config').json()
        assert config['datasets'][0]['available']
        assert config['quality']['billion']['packets']==1_000_000_000
        assert config['quality']['ten_billion']['packets']==10_000_000_000
        assert client.get('/api/datasets/test/cloud').json()['layers'][0]['count']==101
        assert client.get('/api/datasets/test/cloud/secrets').status_code==404
        assert client.post('/api/jobs',json={'dataset_id':'test'},headers={'Origin':'https://elsewhere.example'}).status_code==403
        assert client.post('/api/jobs',json={'dataset_id':'../../tmp'}).status_code==422
        assert client.post('/api/jobs',json={'dataset_id':'test','band':'invalid'}).status_code==422
        assert client.get('/api/config',headers={'Host':'evil.example'}).status_code==400
        assert client.get('/api/jobs/unknown').status_code==404


def test_log_progress_is_phase_specific():
    phase,percent=progress('Starting primary emission...\nLaunched photon packages: 33.5%\n')
    assert phase=='Primary emission' and percent==33.5
    assert progress('Starting secondary emission...')[1] is None
    assert progress('Starting setup...\nIncluding dust emission with equilibrium heating')[0]=='Building galaxy model'


def test_gpu_activation_uses_skirt_log_not_stdout(tmp_path):
    (tmp_path/'console.log').write_text('Starting simulation...\n')
    (tmp_path/'observation_log.txt').write_text('GPU acceleration is using 2 CUDA devices: enabled on NVIDIA\n')
    evidence=execution_evidence(tmp_path)
    assert evidence['gpu_confirmed'] and evidence['execution_log']=='observation_log.txt'


def test_bpass_toddlers_and_weighted_dust_preview(fixture):
    root,catalog=fixture
    text=(root/'base.ski').read_text().replace('BruzualCharlotSEDFamily','BpassSEDFamily')
    text=text.replace('</SourceSystem>','<ParticleSource filename="young.dat"><ToddlersSEDFamily sedMode="SFRNormalized"/></ParticleSource></SourceSystem>')
    (root/'base.ski').write_text(text)
    np.savetxt(root/'young.dat',[[0,0,0,1,.02,.025,320,2]])
    np.savetxt(root/'dust.dat',[[1,0,0,900,0],[2,0,0,900,10],[3,0,0,900,30]])
    directory,meta=catalog.cloud('test')
    assert [x['kind'] for x in meta['layers']]==['stars','young','dust']
    young=meta['layers'][1];dust=meta['layers'][2]
    assert young['weight_total']==2
    assert dust['count']==3 and dust['positive_count']==2 and dust['weight_total']==40
    v=np.fromfile(directory/dust['file'],dtype='<f4').reshape(-1,8)
    v=v[np.argsort(v[:,0])]
    np.testing.assert_allclose(v[1,3:6],v[0,3:6])
    assert v[1,7]/v[0,7]==pytest.approx(np.sqrt(3))
    assert len(np.loadtxt(root/'dust.dat'))==3
    out=root/'out';out.mkdir()
    prepare(catalog.datasets['test'],RenderRequest(dataset_id='test'),out,catalog.cache)
    tree=ET.parse(out/'observation.ski')
    assert tree.find('.//ToddlersSEDFamily') is not None
    assert tree.find('.//BpassSEDFamily') is not None


def test_stellar_kernel_radius_uses_32_other_neighbours():
    reference=np.column_stack((np.arange(41,dtype=float),np.zeros(41),np.zeros(41)))
    radii=neighbour_kernel_radii(reference[[0,20]],reference)
    np.testing.assert_allclose(radii,[32,16])
