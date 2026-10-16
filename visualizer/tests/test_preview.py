import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from observatory.preview import PreviewRequest, cic, reconstruct
from observatory.preview_spectra import StoredTable, interpolate, luminosities, mean_extinction, resource


def test_cic_axes_and_total_mass():
    # Non-symmetric coordinates verify x-fastest WebGL ordering.
    grid = cic(np.array([[1.5, -.5, .5]]), np.array([[3., 5.]]), np.zeros(3), 4., 4)
    np.testing.assert_array_equal(grid[2, 1, 3], [3., 5.])
    np.testing.assert_allclose(grid.sum(axis=(0, 1, 2)), [3., 5.])


@pytest.mark.parametrize('radius', [0., .02, .2, .6])
def test_multiscale_kernel_conserves_light_and_is_positive(radius):
    request = PreviewRequest(size=64, span=8)
    grid = reconstruct(np.array([[.03, -.06, .08]]), np.array([radius]), np.array([[2., 3., 5.]]), request)
    np.testing.assert_allclose(grid.sum(axis=(0, 1, 2), dtype=float)*(8/64)**3, [2, 3, 5], rtol=1e-5)
    assert np.isfinite(grid).all() and np.all(grid >= 0)


def test_kernel_from_outside_cube_can_contribute():
    request = PreviewRequest(size=64, span=4)
    grid = reconstruct(np.array([[2.1, 0., 0.]]), np.array([.8]), np.array([[1.]]), request)
    light = grid.sum(dtype=float)*(4/64)**3
    assert 0 < light < 1


def test_lookup_log_interpolation_and_clipping():
    axes = [np.array([1., 10.]), np.array([2., 20.])]
    values = axes[0][:, None, None]**2*axes[1][None, :, None]*np.array([1., 2., 3.])
    out = interpolate(axes, ['log', 'log'], values, [[np.sqrt(10), np.sqrt(40)], [-1, 100]])
    np.testing.assert_allclose(out[0], 10*np.sqrt(40)*np.array([1., 2., 3.]))
    np.testing.assert_allclose(out[1], [20, 40, 60])


def test_stored_table_layout(tmp_path):
    path = tmp_path/'tiny.stab'
    text = lambda s:s.encode().ljust(8, b' ')
    u64 = lambda n:struct.pack('<Q', n)
    payload = b'SKIRT X\n'+u64(0x010203040A0BFEFF)+u64(2)
    payload += text('lambda')+text('Z')+text('m')+text('1')+text('log')+text('lin')
    for axis in ([1., 2., 3.], [.01, .02]):
        payload += u64(len(axis))+np.asarray(axis, dtype='<f8').tobytes()
    payload += u64(2)+text('Llambda')+text('extra')+text('W/m')+text('1')+text('log')+text('lin')
    expected = np.arange(12.).reshape((2, 3, 2), order='F')
    path.write_bytes(payload+expected.astype('<f8').tobytes(order='F'))
    table = StoredTable(path)
    np.testing.assert_array_equal(table.quantity('Llambda'), expected[0])
    assert table.names == ['lambda', 'Z']


def test_preview_validation():
    for kwargs in ({'size':1024}, {'span':float('nan')}, {'kernel':0}, {'x':1e20}):
        with pytest.raises(ValidationError):
            PreviewRequest(**kwargs)


def test_compact_kernel_defaults_and_api_agree():
    from observatory.app import preview_options
    assert PreviewRequest().kernel == .5
    assert PreviewRequest().size == 256
    assert preview_options() == PreviewRequest()
    assert PreviewRequest(kernel=.1).kernel == .1
    with pytest.raises(ValidationError):
        PreviewRequest(kernel=.09)


@pytest.mark.parametrize('size', [64, 96, 128, 192, 256, 512])
def test_preview_grid_sizes_and_api_agree(size):
    from observatory.app import preview_options
    assert preview_options(size=size) == PreviewRequest(size=size)


def test_finer_grid_reduces_unresolved_source_width_and_conserves_light():
    moments=[]
    for size in (128, 192, 256):
        request=PreviewRequest(size=size, span=60)
        grid=reconstruct(np.zeros((1, 3)), np.array([.07]), np.ones((1, 1)), request)[..., 0]
        dx=request.span/size
        np.testing.assert_allclose(grid.sum(dtype=float)*dx**3, 1., rtol=1e-6)
        # The x marginal measures the numerical width without another 3D array.
        marginal=grid.sum(axis=(0, 1), dtype=float)
        axis=(np.arange(size)+.5-size/2)*dx
        moments.append(float(np.dot(marginal, axis**2)/marginal.sum()))
    np.testing.assert_allclose(np.asarray(moments)/moments[0], [1., (128/192)**2, .25])


def test_large_kernel_memory_guard(monkeypatch):
    from observatory import preview
    monkeypatch.setattr(preview, 'available_memory_bytes', lambda: 64*1024**3)
    preview.check_reconstruction_memory(512, 64, 3, True)
    with pytest.raises(ValueError, match='Choose a smaller volume grid'):
        preview.check_reconstruction_memory(512, 512, 3, True)


def test_memory_guard_runs_before_padded_deposition(monkeypatch):
    from observatory import preview
    monkeypatch.setattr(preview, 'available_memory_bytes', lambda: 1)
    def unexpected_deposition(*args, **kwargs):
        pytest.fail('Should refuse the reconstruction before allocating the padded grid')
    monkeypatch.setattr(preview, 'cic', unexpected_deposition)
    with pytest.raises(ValueError, match='working memory'):
        reconstruct(np.zeros((1, 3)), np.ones(1), np.ones((1, 1)), PreviewRequest(size=64))


def test_smaller_resolved_kernel_sharpens_without_changing_light():
    axis=(np.arange(64)+.5-32)*8/64
    radius_squared=axis[:, None, None]**2+axis[None, :, None]**2+axis[None, None, :]**2
    moments=[]
    for scale in (1., .5):
        grid=reconstruct(np.array([[0.,0.,0.]]), np.array([1.6]), np.array([[1.]]),
                         PreviewRequest(size=64,span=8,kernel=scale))[...,0]
        np.testing.assert_allclose(grid.sum(dtype=float)*(8/64)**3,1.,rtol=1e-5)
        moments.append(float((grid*radius_squared).sum(dtype=float)/grid.sum(dtype=float)))
    assert moments[1] < moments[0]*.4


def test_installed_spectra_and_extinction():
    try:
        resource('BpassSEDFamily_Chabrier100_downsampled')
    except ValueError:
        pytest.skip('SKIRT resource packs are not installed')
    bands = ('SLOAN_SDSS_I', 'SLOAN_SDSS_R', 'SLOAN_SDSS_G')
    params = [[.02, 1e9], [.02, 1e9]]
    output = luminosities('BpassSEDFamily_Chabrier100_downsampled', bands, 0., params, [1., 2.])
    assert np.isfinite(output).all() and np.all(output > 0)
    np.testing.assert_allclose(output[1], 2*output[0])
    assert not np.allclose(output[0], output[0, 0])
    opacity = mean_extinction(bands, 0.)
    assert 0 < opacity[0] < opacity[1] < opacity[2]
    shifted = luminosities('BpassSEDFamily_Chabrier100_downsampled', bands, 1., params, [1., 2.])
    assert not np.allclose(output, shifted)


def test_preview_api_guards(tmp_path):
    from observatory.app import create_app
    config = tmp_path/'datasets.json'; config.write_text(json.dumps({'datasets':[]}))
    app = create_app(config=config, storage=tmp_path/'storage')
    with TestClient(app) as client:
        assert client.get('/api/datasets/missing/preview-volume').status_code == 404
        assert client.get('/api/datasets/missing/preview-volume?size=64').status_code == 404
        for size in (192, 256, 512):
            assert client.get(f'/api/datasets/missing/preview-volume?size={size}').status_code == 404
        assert client.get('/api/datasets/missing/preview-volume?size=1024').status_code == 422
        assert client.get('/api/preview-volumes/not-a-key/stars-dust.bin').status_code == 404
        assert client.get('/api/preview-volumes/'+'a'*64+'/unknown.bin').status_code == 404
