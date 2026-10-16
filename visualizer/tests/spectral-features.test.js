import test from 'node:test';
import assert from 'node:assert/strict';
import {SPECTRAL_FEATURES,FEATURE_GROUPS,referenceFeatures,placeFeatureLabels} from '../frontend/spectral-features.js';

const all=new Set(Object.keys(FEATURE_GROUPS));
test('Feature references are unique, finite and sourced',()=>{
  assert.equal(new Set(SPECTRAL_FEATURES.map(f=>f.id)).size,SPECTRAL_FEATURES.length);
  for(const f of SPECTRAL_FEATURES){assert(f.rest>0&&Number.isFinite(f.rest));assert(all.has(f.group));assert(f.meaning&&f.source.startsWith('https://'));}
});
test('Observer references use the saved redshift; rest spectra never shift',()=>{
  const observed=referenceFeatures({spectral_frame:'observer',reference_redshift:1},all,[-2,2]);
  assert.equal(observed.find(f=>f.id==='halpha').wavelength,.656461*2);
  assert.equal(observed.find(f=>f.id==='neii').wavelength,12.81355*2);
  const rest=referenceFeatures({spectral_frame:'rest',reference_redshift:5},all,[-2,2]);
  assert.equal(rest.find(f=>f.id==='halpha').wavelength,.656461);
  for(const z of [undefined,null,NaN,Infinity,-1,'1'])assert.deepEqual(referenceFeatures({spectral_frame:'observer',reference_redshift:z},all,[-2,2]),[]);
});
test('Group and zoom filters exclude references outside the displayed spectrum',()=>{
  const selected=referenceFeatures({spectral_frame:'rest'},new Set(['pah']),[Math.log10(5),Math.log10(12)]);
  assert(selected.length>0);assert(selected.every(f=>f.group==='pah'&&f.wavelength>=5&&f.wavelength<=12));
  assert.deepEqual(referenceFeatures({spectral_frame:'rest'},new Set(),[-2,2]),[]);
});
test('Desktop and mobile labels remain inside the plot without same-lane collisions',()=>{
  const features=referenceFeatures({spectral_frame:'rest'},all,[-1,2]);
  for(const width of [290,390,900,1200]){
    const left=72,right=width-18,sx=w=>left+(Math.log10(w)+1)/3*(right-left);
    const labels=placeFeatureLabels(features,sx,left,right);
    assert(labels.length>0);
    for(const a of labels){
      assert(a.x-a.width/2>=left-1e-9&&a.x+a.width/2<=right+1e-9);
      for(const b of labels)if(a!==b&&a.lane===b.lane)assert(Math.abs(a.x-b.x)>(a.width+b.width)/2);
    }
  }
});
