"""Read-only morphology comparison against an existing SKIRT RGB FITS.

Run: PYTHONPATH=. .venv/bin/python tests/compare_preview.py JOB_ID
This CPU reference mirrors the browser's volume integration; it is not a
GPU benchmark or a replacement for the browser shader tests.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

from observatory.catalog import Catalog
from observatory.preview import PreviewRequest, PreviewVolumes
from observatory.simulation import BANDS, Camera, camera_basis


def integrate(fields, meta, camera, pixels=128, steps=384):
    right, up, observer = camera_basis(camera)
    axis = (np.arange(pixels)+.5-pixels/2)*camera.fov_kpc/pixels
    xx, yy = np.meshgrid(axis, axis)
    origins = (np.asarray(camera.target_kpc)+xx[..., None]*right+yy[..., None]*up+observer*meta['request']['span']*3).reshape(-1, 3)
    ray = -observer
    safe = np.where(np.abs(ray) < 1e-10, 1e-10, ray)
    lower, upper = np.array(meta['bounds_kpc'])
    a, b = (lower-origins)/safe, (upper-origins)/safe
    enter, leave = np.maximum(np.minimum(a,b).max(axis=1),0), np.maximum(a,b).min(axis=1)
    ds = np.maximum(leave-enter,0)/steps
    light = np.zeros((len(origins),3)); transmittance=np.ones_like(light)
    for step in range(steps):
        point = origins+(enter+(step+.5)*ds)[:,None]*ray
        coordinates = ((point-lower)/meta['voxel_kpc']-.5).T[::-1]
        values = np.column_stack([map_coordinates(fields[..., channel],coordinates,order=1,mode='nearest') for channel in range(7)])
        tau = values[:,3,None]*np.asarray(meta['extinction_per_kpc'])*ds[:,None]
        factor = -np.expm1(-tau)/np.maximum(tau,1e-20)
        factor[tau < 1e-6] = 1-tau[tau < 1e-6]/2
        light += transmittance*(values[:,:3]+values[:,4:7])*ds[:,None]*factor
        transmittance *= np.exp(-tau)
    return light.reshape(pixels,pixels,3)


def main():
    identifier=sys.argv[1]
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Expected an existing job id')
    folder=Path('var/jobs')/identifier
    job=json.loads((folder/'job.json').read_text()); manifest=json.loads((folder/'manifest.json').read_text())
    request=job['request']; camera=Camera(**request['camera']); bands=request['rgb_bands']
    catalog=Catalog(Path('datasets.json'),Path('var/catalogues'))
    dataset=catalog.datasets[request['dataset_id']]
    current=dataset.fingerprint()[1]
    provenance_matches=all(current.get(key)==manifest['provenance'].get(key) for key in ('base_sha256','inputs'))
    # Keep the original comparison settings explicit as UI defaults evolve.
    options=PreviewRequest(**bands,size=128,span=dataset.info()['fov_kpc'],kernel=1)
    previews=PreviewVolumes(catalog); meta=previews.build(dataset.id,options)
    volume_folder=previews.root/meta['key']; n=meta['size']
    stars=np.fromfile(volume_folder/'stars-dust.bin',dtype='<f4').reshape(n,n,n,4)
    young=np.fromfile(volume_folder/'young.bin',dtype='<f4').reshape(n,n,n,4)
    predicted=integrate(np.concatenate((stars,young[...,:3]),axis=-1),meta,camera)
    source=folder/'observation_rgb_total.fits'; digest=hashlib.sha256(source.read_bytes()).hexdigest()
    cube=fits.getdata(source).astype(float); scale=cube.shape[1]//128
    cube=cube.reshape(3,128,scale,128,scale).mean(axis=(2,4))
    order=sorted(bands.values(),key=lambda band:BANDS[band]['pivot_micron'])
    report={}
    for channel,color in enumerate(('red','green','blue')):
        reference=cube[order.index(bands[color])].ravel(); preview=predicted[...,channel].ravel()
        reference/=max(reference.sum(),1e-100); preview/=max(preview.sum(),1e-100)
        report[color]={'band':bands[color], 'linear_morphology_correlation':float(np.corrcoef(reference,preview)[0,1]),
                       'normalized_l1_difference':float(np.abs(reference-preview).sum())}
    print(json.dumps(dict(job=identifier,source_fits_sha256=digest,source_provenance_matches=provenance_matches,
                          camera=request['camera'],volume_key=meta['key'],voxel_kpc=meta['voxel_kpc'],channels=report,
                          note='Each channel is normalized by total flux for morphology only. No parity, convergence, or timing claim.'),indent=2))


if __name__=='__main__':
    main()
