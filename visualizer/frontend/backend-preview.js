import * as THREE from 'three';

export function backendDimensions(width,height,scale=1){
  const factor=Math.min(scale,1920/Math.max(width,height),Math.sqrt(3_600_000/(width*height)));
  return {width:Math.max(1,Math.floor(width*factor)),height:Math.max(1,Math.floor(height*factor))};
}

export function unpackBackendFrame(buffer){
  if(buffer.byteLength<4)throw new Error('Truncated backend frame');
  const length=new DataView(buffer).getUint32(0,true);
  if(length>65536||4+length>buffer.byteLength)throw new Error('Invalid backend frame header');
  const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,length)));
  let offset=4+length;
  const images=meta.views.map(view=>{
    if(!Number.isSafeInteger(view.bytes)||view.bytes<0||offset+view.bytes>buffer.byteLength)throw new Error('Invalid backend image length');
    const bytes=new Uint8Array(buffer,offset,view.bytes);offset+=view.bytes;
    return {view,bytes};
  });
  if(offset!==buffer.byteLength)throw new Error('Unexpected backend frame data');
  return {meta,images};
}

async function responseError(response){
  let body;try{body=await response.json();}catch{body={detail:response.statusText};}
  return new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));
}

export class BackendVolumePreview{
  static async create(meta,signal,accessKey){
    const response=await fetch('/api/backend-preview/sessions',{method:'POST',signal,
      headers:accessKey?{Authorization:`Bearer ${accessKey}`}:{}});
    if(!response.ok)throw await responseError(response);
    const session=await response.json();
    const preview=new BackendVolumePreview(meta,session);
    preview.accessKey=accessKey;
    if(signal.aborted){preview.dispose();throw new DOMException('Aborted','AbortError');}
    return preview;
  }
  constructor(meta,session){
    this.meta=meta;this.session=session;this.remote=true;this.generation=0;this.sequence=0;
    this.settled=false;this.busy=false;this.disposed=false;this.failed=false;this.retryAt=0;
    this.signature='';this.textures=new Map();this.changed=performance.now();this.hasUpdated=false;
    this.material=new THREE.MeshBasicMaterial({depthTest:false,depthWrite:false,toneMapped:false});
    this.scene=new THREE.Scene();this.quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material);
    this.scene.add(this.quad);this.camera=new THREE.Camera();
  }
  reset(){this.generation++;this.changed=performance.now();this.settled=false;}
  get interactive(){return performance.now()-this.changed<180;}
  get needsFrame(){return !this.failed&&(!this.settled||this.interactive||this.busy||(this.wasInteractive&&!this.interactive));}
  async update(payload,notify){
    if(this.disposed||this.failed)return;
    this.wasInteractive=this.interactive;
    const signature=JSON.stringify(payload);
    if(signature!==this.signature){this.signature=signature;this.generation++;this.settled=false;}
    if(this.busy||this.settled||performance.now()<this.retryAt)return;
    this.busy=true;const generation=this.generation,sequence=++this.sequence;
    this.controller=new AbortController();
    try{
      const response=await fetch('/api/backend-preview/frame',{method:'POST',signal:this.controller.signal,
        headers:{'Content-Type':'application/json',Authorization:`Bearer ${this.session.token}`},
        body:JSON.stringify({...payload,sequence,session_id:this.session.session_id,volume_key:this.meta.key})});
      if(response.status===429){this.retryAt=performance.now()+250;return;}
      if(response.status===401){
        const renewed=await fetch('/api/backend-preview/sessions',{method:'POST',signal:this.controller.signal,
          headers:this.accessKey?{Authorization:`Bearer ${this.accessKey}`}:{}});
        if(!renewed.ok)throw await responseError(renewed);
        this.session=await renewed.json();this.sequence=0;this.generation++;this.settled=false;
        return;
      }
      if(!response.ok)throw await responseError(response);
      const {meta,images}=unpackBackendFrame(await response.arrayBuffer());
      if(this.disposed||generation!==this.generation||meta.sequence!==sequence)return;
      const decoded=[];
      try{
        for(const {view,bytes} of images)decoded.push({view,bitmap:await createImageBitmap(new Blob([bytes],{type:'image/png'}),{imageOrientation:'flipY',premultiplyAlpha:'none'})});
        if(this.disposed||generation!==this.generation)return;
        for(const {view,bitmap} of decoded){
          const previous=this.textures.get(view.name);previous?.dispose();previous?.image.close();
          const texture=new THREE.Texture(bitmap);texture.colorSpace=THREE.SRGBColorSpace;
          texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.generateMipmaps=false;texture.needsUpdate=true;
          this.textures.set(view.name,texture);
        }
        this.settled=meta.views.every(view=>view.samples>=8);this.hasUpdated=true;this.lastMeta=meta;
        notify(null,meta);
      }finally{
        for(const {bitmap} of decoded)if(![...this.textures.values()].some(texture=>texture.image===bitmap))bitmap.close();
      }
    }catch(error){
      if(!this.disposed&&generation===this.generation&&error.name!=='AbortError'){this.failed=true;notify(error);}
    }finally{this.busy=false;}
  }
  render(renderer,name){
    renderer.setRenderTarget(null);renderer.setClearColor(0,1);renderer.clear();
    const texture=this.textures.get(name);
    if(!texture)return;
    if(this.material.map!==texture){this.material.map=texture;this.material.needsUpdate=true;}
    renderer.render(this.scene,this.camera);
    renderer.domElement.dataset.volumeSamples=String(this.lastMeta?.views.find(view=>view.name===name)?.samples||0);
  }
  dispose(){
    this.disposed=true;this.generation++;this.controller?.abort();
    fetch(`/api/backend-preview/sessions/${this.session.session_id}`,{method:'DELETE',keepalive:true,
      headers:{Authorization:`Bearer ${this.session.token}`}}).catch(()=>{});
    for(const texture of this.textures.values()){texture.dispose();texture.image.close();}
    this.textures.clear();this.material.dispose();this.quad.geometry.dispose();
  }
}
