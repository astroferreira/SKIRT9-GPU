import test from 'node:test';
import assert from 'node:assert/strict';
import {BackendVolumePreview,backendDimensions,unpackBackendFrame} from '../frontend/backend-preview.js';
import {displayExposure} from '../frontend/viewer.js';

test('Exposure floor is extended without changing the default mapping',()=>{
  assert.ok(Math.abs(displayExposure(-2)-2e-7)<1e-20);
  assert.equal(displayExposure(.38),.002*Math.pow(100,.38));
  assert.ok(displayExposure(.01)/displayExposure(-2)>10000);
});

test('Backend output dimensions are bounded and retain aspect ratio',()=>{
  assert.deepEqual(backendDimensions(1000,800,.5),{width:500,height:400});
  assert.deepEqual(backendDimensions(3840,2160),{width:1920,height:1080});
  const large=backendDimensions(4000,4000);assert.ok(large.width*large.height<=3600000);
});

test('Backend packets preserve image boundaries and reject truncation',()=>{
  const header=new TextEncoder().encode(JSON.stringify({views:[{name:'main',bytes:3},{name:'mini',bytes:2}]}));
  const buffer=new ArrayBuffer(4+header.length+5);
  new DataView(buffer).setUint32(0,header.length,true);new Uint8Array(buffer,4,header.length).set(header);
  new Uint8Array(buffer,4+header.length).set([1,2,3,4,5]);
  const result=unpackBackendFrame(buffer);
  assert.deepEqual([...result.images[0].bytes],[1,2,3]);assert.deepEqual([...result.images[1].bytes],[4,5]);
  assert.throws(()=>unpackBackendFrame(buffer.slice(0,-1)));
  assert.throws(()=>unpackBackendFrame(new ArrayBuffer(2)));
});

test('Expired backend sessions reconnect without switching rendering mode',async t=>{
  t.mock.method(globalThis,'fetch',async url=>url.endsWith('/frame')?new Response('{}',{status:401}):
    new Response(JSON.stringify({session_id:'renewed',token:'new-token'})));
  const preview=new BackendVolumePreview({key:'volume'},{session_id:'old',token:'old-token'});
  let error=null;await preview.update({},value=>{error=value;});
  assert.equal(error,null);assert.equal(preview.failed,false);assert.equal(preview.session.session_id,'renewed');
  assert.equal(preview.sequence,0);assert.equal(preview.settled,false);preview.dispose();
});

test('Frames from obsolete camera revisions are discarded',async t=>{
  let resolve;
  t.mock.method(globalThis,'fetch',url=>url.endsWith('/frame')?new Promise(done=>{resolve=done;}):Promise.resolve(new Response('{}')));
  const preview=new BackendVolumePreview({key:'volume'},{session_id:'old',token:'token'});
  let notifications=0;const pending=preview.update({},()=>notifications++);
  preview.reset();
  const header=new TextEncoder().encode(JSON.stringify({sequence:1,views:[]}));
  const packet=new Uint8Array(4+header.length);new DataView(packet.buffer).setUint32(0,header.length,true);packet.set(header,4);
  resolve(new Response(packet));await pending;
  assert.equal(notifications,0);assert.equal(preview.lastMeta,undefined);assert.equal(preview.busy,false);preview.dispose();
});
