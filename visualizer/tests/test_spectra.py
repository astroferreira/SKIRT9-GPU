import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient
from pydantic import ValidationError

from observatory.app import create_app
from observatory.jobs import Jobs
from observatory.simulation import Camera, RenderRequest, camera_basis, prepare
from observatory.spectra import SPECTRAL_MODES, expected_bins, process_sed, smooth_spectrum, resolved_spectral_mode, reference_redshift
from test_pipeline import fixture, wait_for


def test_reference_redshift_uses_saved_observation_and_fails_closed(tmp_path):
    path=tmp_path/'observation.ski'
    assert reference_redshift(path,'observer') is None
    assert reference_redshift(path,'rest')==0
    path.write_text('<root><FlatUniverseCosmology redshift="1.5"/></root>')
    assert reference_redshift(path,'observer')==1.5
    assert reference_redshift(path,'rest')==0
    path.write_text('<root><LocalUniverseCosmology/></root>')
    assert reference_redshift(path,'observer')==0
    for value in ['nan','inf','-1','unknown']:
        path.write_text(f'<root><FlatUniverseCosmology redshift="{value}"/></root>')
        assert reference_redshift(path,'observer') is None


@pytest.mark.parametrize('mode_id', list(SPECTRAL_MODES))
def test_sed_run_contains_only_requested_observer_and_preserves_physics(fixture, mode_id):
    root,catalog=fixture
    out=root/'sed';out.mkdir()
    camera=Camera(inclination=73,azimuth=-45,roll=37,target_kpc=(9,12,-3),fov_kpc=8)
    request=RenderRequest(dataset_id='test',output='sed',spectral_mode=mode_id,camera=camera)
    original=ET.parse(root/'base.ski')
    original.find('.//MonteCarloSimulation').set('iterateSecondaryEmission','true')
    original.write(root/'base.ski')
    manifest=prepare(catalog.datasets['test'],request,out,catalog.cache)
    result=ET.parse(out/'observation.ski')
    instruments=result.find('.//InstrumentSystem/instruments')
    assert len(instruments)==1 and instruments[0].tag=='SEDInstrument'
    sed=instruments[0]
    assert sed.get('radius')=='0 pc' and sed.get('inclination')=='73 deg' and sed.get('azimuth')=='-45 deg'
    assert sed.get('centerX') is None and sed.get('fieldOfViewX') is None
    assert result.find('.//FrameInstrument') is None and result.find('.//BroadBand') is None
    mode=SPECTRAL_MODES[mode_id]
    grid=sed.find('wavelengthGrid/ResolutionWavelengthGrid')
    assert float(grid.get('resolution'))==2*mode['resolving_power']
    assert float(grid.get('minWavelength').split()[0])==mode['min_micron']
    assert float(grid.get('maxWavelength').split()[0])==mode['max_micron']
    def physics(element):
        return [(child.tag,{key:value for key,value in child.attrib.items() if key!='filename'}) for child in element.iter()]
    for tag in ('SourceSystem','MediumSystem','FlatUniverseCosmology'):
        assert physics(result.find(f'.//{tag}'))==physics(original.find(f'.//{tag}'))
    assert result.find('.//MonteCarloSimulation').get('iterateSecondaryEmission')=='true'
    assert result.find('.//ExtragalacticUnits').get('fluxOutputStyle')=='Neutral'
    assert manifest['spectral_observation']['frame']=='observer'
    assert manifest['spectral_observation']['bins']==expected_bins(mode)
    assert sed.get('roll')=='0 deg'
    right,up,observer=camera_basis(camera.model_copy(update={'roll':0}))
    for key,expected in [('right',right),('up',up),('observer',observer)]:
        np.testing.assert_allclose(manifest['projection'][key],expected)


def test_sed_preserves_fixed_distance_and_reports_rest_frame(fixture):
    root,catalog=fixture
    original=ET.parse(root/'base.ski')
    original.find('.//FrameInstrument').set('distance','12 Mpc')
    original.write(root/'base.ski')
    out=root/'sed';out.mkdir()
    manifest=prepare(catalog.datasets['test'],RenderRequest(dataset_id='test',output='sed'),out,catalog.cache)
    assert ET.parse(out/'observation.ski').find('.//SEDInstrument').get('distance')=='12 Mpc'
    assert manifest['spectral_observation']['frame']=='rest'


def write_sed(path, mode_id='nirspec_prism', blank=False, resolving_power=None):
    mode=resolved_spectral_mode(mode_id,resolving_power)
    wave=np.geomspace(mode['min_micron'],mode['max_micron'],expected_bins(mode))
    flux=np.zeros_like(wave) if blank else 1e-15*(1+np.exp(-((np.log(wave)-.7)/.015)**2))
    np.savetxt(path,np.column_stack((wave,flux)),fmt='%.12e',header='SED test fixture\ncolumn 1: wavelength; lambda (micron)\ncolumn 2: total flux; lambda*F_lambda (W/m2)')
    return wave,flux


@pytest.mark.parametrize('mode_id',['nirspec_prism','miri_mrs'])
def test_spectral_products_keep_raw_data_units_and_frame(tmp_path, mode_id):
    path=tmp_path/'raw.dat';wave,flux=write_sed(path,mode_id)
    before=path.read_bytes()
    meta=process_sed(path,tmp_path,mode_id,'Test galaxy','observer')
    assert path.read_bytes()==before
    assert meta['samples']==len(wave) and meta['positive_bins']==len(wave)
    data=json.loads((tmp_path/'spectrum.json').read_text())
    np.testing.assert_allclose(data['wavelength_micron'],wave,rtol=1e-12)
    np.testing.assert_allclose(data['raw_flux'],flux,rtol=1e-12)
    with fits.open(tmp_path/'spectrum.fits') as hdus:
        assert hdus[0].header['SPECFRAM']=='observer'
        assert hdus[0].header['APERTURE']=='ALL'
        assert 'SED test fixture' in str(hdus[0].header['HISTORY'])
        np.testing.assert_array_equal(hdus[1].data['LAMBDA_FLAMBDA_RAW'],data['raw_flux'])
        assert hdus[1].columns[0].unit=='um' and hdus[1].columns[1].unit=='W m-2'
    csv=np.loadtxt(tmp_path/'spectrum.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(csv[:,2],data['smoothed_flux'],rtol=1e-12)
    assert (tmp_path/'spectrum.png').stat().st_size>1000


def test_gaussian_convolution_preserves_constant_flux_and_line_area():
    wave=np.geomspace(.6,5.3,expected_bins(SPECTRAL_MODES['nirspec_prism']))
    np.testing.assert_allclose(smooth_spectrum(wave,np.ones_like(wave),100),1,rtol=1e-14)
    raw=np.zeros_like(wave);raw[len(raw)//2]=1
    convolved=smooth_spectrum(wave,raw,100)
    assert 0<convolved.max()<1
    assert convolved.sum()==pytest.approx(1,rel=1e-14)


def test_blank_spectrum_is_reported_and_bad_units_rejected(tmp_path):
    path=tmp_path/'raw.dat';write_sed(path,blank=True)
    assert process_sed(path,tmp_path,'nirspec_prism','Test','rest')['blank']
    path.write_text(path.read_text().replace('(W/m2)','(Jy)'))
    with pytest.raises(ValueError,match='W/m2'):
        process_sed(path,tmp_path,'nirspec_prism','Test','rest')


@pytest.mark.parametrize('bad',[float('nan'),-1])
def test_invalid_spectral_flux_is_rejected(tmp_path,bad):
    path=tmp_path/'raw.dat';write_sed(path)
    lines=path.read_text().splitlines();lines[-1]=f'5.3 {bad}'
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='invalid or negative'):
        process_sed(path,tmp_path,'nirspec_prism','Test','rest')


def test_sed_cache_ignores_image_only_settings_but_keeps_direction_and_mode():
    a=RenderRequest(dataset_id='test',output='sed')
    b=RenderRequest(dataset_id='test',output='sed',pixels=1024,band='SLOAN_SDSS_U',camera=Camera(roll=90,fov_kpc=5,target_kpc=(3,9,-5)))
    assert a.render_identity()==b.render_identity()
    assert a.render_identity()!=a.model_copy(update={'spectral_mode':'miri_mrs'}).render_identity()
    assert a.render_identity()!=a.model_copy(update={'camera':Camera(azimuth=55)}).render_identity()
    image=RenderRequest(dataset_id='test')
    assert 'output' not in image.render_identity() and 'spectral_mode' not in image.render_identity()
    assert image.render_identity()!=a.render_identity()
    with pytest.raises(ValidationError):RenderRequest(dataset_id='test',spectral_mode='invalid')


@pytest.mark.parametrize('resolution',[10,750,10000])
def test_custom_sed_resolution_preserves_coverage_and_physics(fixture,resolution):
    root,catalog=fixture
    out=root/'custom-sed';out.mkdir()
    request=RenderRequest(dataset_id='test',output='sed',spectral_mode='miri_mrs',spectral_resolution=resolution)
    manifest=prepare(catalog.datasets['test'],request,out,catalog.cache)
    tree=ET.parse(out/'observation.ski')
    grid=tree.find('.//SEDInstrument/wavelengthGrid/ResolutionWavelengthGrid')
    assert float(grid.get('resolution'))==2*resolution
    assert grid.get('minWavelength')=='4.9 micron'
    assert tree.find('.//FrameInstrument') is None
    assert tree.find('.//DustEmissionOptions').get('dustEmissionType')=='Stochastic'
    assert manifest['spectral_observation']['resolving_power']==resolution
    assert manifest['spectral_observation']['bins']==expected_bins(resolved_spectral_mode('miri_mrs',resolution))
    assert SPECTRAL_MODES['miri_mrs']['resolving_power']==3000


@pytest.mark.parametrize('bad',[0,9,10001,-1,750.5,True,'750',float('nan'),float('inf')])
def test_custom_resolution_rejects_invalid_values(bad):
    with pytest.raises(ValidationError):RenderRequest(dataset_id='test',output='sed',spectral_resolution=bad)


def test_resolution_cache_compatibility():
    old=RenderRequest(dataset_id='test',output='sed',spectral_mode='miri_mrs')
    explicit=old.model_copy(update={'spectral_resolution':3000})
    assert old.render_identity()==explicit.render_identity()
    assert 'spectral_resolution' not in old.render_identity()
    assert old.render_identity()!=old.model_copy(update={'spectral_resolution':750}).render_identity()
    image=RenderRequest(dataset_id='test')
    assert image.render_identity()==image.model_copy(update={'spectral_resolution':750}).render_identity()


def test_custom_resolution_reaches_all_spectral_products(tmp_path):
    path=tmp_path/'raw.dat';wave,flux=write_sed(path,'miri_mrs',resolving_power=750)
    before=path.read_bytes()
    metadata=process_sed(path,tmp_path,'miri_mrs','Test halo','observer',resolving_power=750)
    assert path.read_bytes()==before
    assert metadata['samples']==len(wave)
    assert metadata['mode']['resolving_power']==750
    assert metadata['smoothing']['resolving_power']==750
    data=json.loads((tmp_path/'spectrum.json').read_text())
    np.testing.assert_allclose(data['raw_flux'],flux,rtol=1e-12)
    with fits.open(tmp_path/'spectrum.fits') as hdus:
        assert hdus[0].header['RESPOWER']==750
        assert len(hdus[1].data)==len(wave)


def fake_sed_worker(root):
    binary=root/'fake-sed-worker'
    binary.write_text(f'''#!{sys.executable}
import math,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
folder=Path(sys.argv[sys.argv.index('-o')+1])
tree=ET.parse(folder/'observation.ski')
assert tree.find('.//FrameInstrument') is None
grid=tree.find('.//SEDInstrument/wavelengthGrid/ResolutionWavelengthGrid')
low=float(grid.get('minWavelength').split()[0]);high=float(grid.get('maxWavelength').split()[0]);r=float(grid.get('resolution'))
n=1+math.ceil(math.log(high/low)/math.log1p(1/r))
np.savetxt(folder/'observation_spectrum_sed.dat',np.column_stack((np.geomspace(low,high,n),np.ones(n)*1e-15)),header='column 1: wavelength; lambda (micron)\\ncolumn 2: total flux; lambda*F_lambda (W/m2)')
print('TEST ONLY: fake SED subprocess')
''')
    binary.chmod(0o755);return binary


@pytest.mark.parametrize('resolution',[None,750])
def test_sed_job_api_downloads_cache_and_history(fixture,resolution):
    root,catalog=fixture
    binary=fake_sed_worker(root)
    app=create_app(root/'datasets.json',root/'state',binary)
    with TestClient(app) as client:
        config=client.get('/api/config').json()
        assert 'miri_mrs' in config['spectral_modes'] and len(config['spectral_modes'])==11
        result=client.post('/api/jobs',json={'dataset_id':'test','output':'sed','spectral_mode':'miri_lrs','spectral_resolution':resolution})
        assert result.status_code==202
        job=result.json()['job'];identifier=job['id']
        finished=wait_for(app.state.jobs,identifier,{'completed','failed'},timeout=20)
        assert finished['status']=='completed',finished
        assert finished['gpu_confirmed'] is False
        assert finished['spectrum']['mode']['resolving_power']==(resolution or 100)
        assert finished['request']['spectral_resolution']==resolution
        assert 'image_url' not in finished and finished['spectrum']['mode_id']=='miri_lrs'
        for key in ('preview_url','fits_url','csv_url','raw_url','spectrum_url'):
            assert client.get(finished[key]).status_code==200
        assert client.get(finished['spectrum_url']).json()['flux_quantity']=='lambda*F_lambda'
        stored=root/'state/jobs'/identifier/'spectrum.json'
        before=stored.read_bytes()
        assert client.get(finished['spectrum_url']).json()['reference_redshift']==1.
        assert stored.read_bytes()==before
        assert client.get(f'/api/jobs/{identifier}/files/observation_view_total.fits').status_code==404
        cached=client.post('/api/jobs',json={**job['request'],'pixels':1024}).json()
        assert cached['reused'] and cached['job']['id']==identifier
        assert client.post('/api/jobs',json={'dataset_id':'test','output':'sed','spectral_mode':'fake'}).status_code==422
    # The spectrum remains downloadable and reusable after server restart.
    with TestClient(create_app(root/'datasets.json',root/'state',binary)) as client:
        assert client.get(f'/api/jobs/{identifier}').json()['status']=='completed'
        assert client.get(f'/api/jobs/{identifier}/spectrum').status_code==200
