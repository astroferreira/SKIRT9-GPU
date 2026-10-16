import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { cameraBasis, preset } from '../frontend/camera.js';

test('Camera matches SKIRT detector axes and screen handedness for arbitrary angles',()=>{
  for(const inclination of [0,17,90,134,180])for(const azimuth of [-180,-67,0,90,180])for(const roll of [-179,-25,0,90]){
    const b=cameraBasis({inclination,azimuth,roll});
    const right=new THREE.Vector3(...b.right),up=new THREE.Vector3(...b.up),observer=new THREE.Vector3(...b.observer);
    assert.ok(right.clone().cross(up).distanceTo(observer)<1e-12);
    const camera=new THREE.OrthographicCamera(-20,20,20,-20,.01,1000);
    const target=new THREE.Vector3(4,-7,3);
    camera.position.copy(target).addScaledVector(observer,100);camera.up.copy(up);camera.lookAt(target);camera.updateMatrixWorld();
    const projected=target.clone().addScaledVector(right,5).addScaledVector(up,10).project(camera);
    assert.ok(Math.abs(projected.x-.25)<1e-12);
    assert.ok(Math.abs(projected.y-.5)<1e-12);
  }
});
test('Face-on XY preset puts model X right and Y up',()=>{
  const b=cameraBasis(preset('xy',60));
  assert.ok(Math.abs(b.right[0]-1)<1e-12);assert.ok(Math.abs(b.up[1]-1)<1e-12);
});
