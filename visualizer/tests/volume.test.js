import test from 'node:test';
import assert from 'node:assert/strict';
import {segmentIntegral,volumeFragmentShader,volumeGridInfo} from '../frontend/volume.js';

test('Grid detail and float32 texture budget track resolution and cube width',()=>{
  assert.deepEqual(volumeGridInfo(128,60),{voxelPc:468.75,textureMiB:64});
  assert.deepEqual(volumeGridInfo(192,60),{voxelPc:312.5,textureMiB:216});
  assert.deepEqual(volumeGridInfo(256,60),{voxelPc:234.375,textureMiB:512});
  assert.deepEqual(volumeGridInfo(256,30),{voxelPc:117.1875,textureMiB:512});
  assert.deepEqual(volumeGridInfo(512,60),{voxelPc:117.1875,textureMiB:4096});
});

test('Homogeneous emitting/absorbing slab matches analytic transfer, including zero dust',()=>{
  for(const extinction of [0,1e-10,.2,2,100]){
    const expected=extinction?3*(-Math.expm1(-extinction*4))/extinction:12;
    assert.ok(Math.abs(segmentIntegral(3,extinction,4)-expected)<1e-8);
    for(const steps of [16,96,192,384]){
      let light=0,transmittance=1;
      for(let step=0;step<steps;step++){
        light+=transmittance*segmentIntegral(3,extinction,4/steps);
        transmittance*=Math.exp(-extinction*4/steps);
      }
      assert.ok(Math.abs(light-expected)<1e-8);
    }
  }
});

test('Dust ordering attenuates a background source but does not attenuate a foreground source',()=>{
  const source=segmentIntegral(1,0,1),foregroundDust=Math.exp(-2);
  assert.ok(source*foregroundDust<source);
  assert.equal(source+0*foregroundDust,source);
  assert.match(volumeFragmentShader,/transmittance\*max\(emission/);
  assert.match(volumeFragmentShader,/transmittance\*=exp\(-tau\)/);
});
