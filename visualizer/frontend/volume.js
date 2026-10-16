import * as THREE from 'three';

export function volumeGridInfo(size, span) {
  // Two RGBA float32 volumes. Each WebGL context uploads its own texture copy.
  return {voxelPc:1000*span/size,textureMiB:2*4*4*size**3/1024**2};
}

export const volumeVertexShader = `
  out vec2 vUv;
  void main() { vUv=uv; gl_Position=vec4(position.xy,0.,1.); }
`;

export const volumeFragmentShader = `
  precision highp float;
  precision highp sampler3D;
  uniform sampler3D starsDust;
  uniform sampler3D youngSources;
  uniform sampler2D history;
  uniform mat4 inverseProjection;
  uniform mat4 cameraWorld;
  uniform vec3 boxMin;
  uniform vec3 boxMax;
  uniform vec3 extinction;
  uniform vec2 resolution;
  uniform vec2 pixelJitter;
  uniform float depthJitter;
  uniform float dustScale;
  uniform float showStars;
  uniform float showYoung;
  uniform int steps;
  uniform int sampleIndex;
  in vec2 vUv;
  out vec4 outColor;

  vec3 unproject(vec2 xy,float z) {
    vec4 p=inverseProjection*vec4(xy,z,1.);
    return (cameraWorld*vec4(p.xyz/p.w,1.)).xyz;
  }
  // Integral of a constant emitting/absorbing segment, stable at zero dust.
  vec3 segmentFactor(vec3 tau) {
    vec3 exact=(1.-exp(-tau))/max(tau,vec3(1e-6));
    return mix(exact,1.-tau*.5+tau*tau/6.,lessThan(tau,vec3(.001)));
  }
  void main() {
    vec2 ndc=(vUv+pixelJitter/resolution)*2.-1.;
    vec3 origin=unproject(ndc,-1.);
    vec3 farPoint=unproject(ndc,1.);
    vec3 ray=normalize(farPoint-origin);
    vec3 safe=mix(vec3(1e-10),ray,greaterThan(abs(ray),vec3(1e-10)));
    vec3 a=(boxMin-origin)/safe, b=(boxMax-origin)/safe;
    vec3 lower=min(a,b), upper=max(a,b);
    float enter=max(0.,max(lower.x,max(lower.y,lower.z)));
    float leave=min(length(farPoint-origin),min(upper.x,min(upper.y,upper.z)));
    vec3 light=vec3(0.), transmittance=vec3(1.);
    if (leave>enter) {
      float ds=(leave-enter)/float(steps);
      for(int i=0;i<512;i++) {
        if(i>=steps) break;
        vec3 point=origin+ray*(enter+(float(i)+depthJitter)*ds);
        vec3 uvw=(point-boxMin)/(boxMax-boxMin);
        vec4 stellar=texture(starsDust,uvw);
        vec3 emission=stellar.rgb*showStars+texture(youngSources,uvw).rgb*showYoung;
        vec3 tau=max(stellar.a,0.)*extinction*dustScale*ds;
        light+=transmittance*max(emission,vec3(0.))*ds*segmentFactor(tau);
        transmittance*=exp(-tau);
        if(max(transmittance.r,max(transmittance.g,transmittance.b))<1e-5) break;
      }
    }
    // Accumulate linear radiance, before exposure, PSF, and display stretch.
    vec3 previous=texture(history,vUv).rgb;
    outColor=vec4(mix(previous,light,1./float(sampleIndex+1)),1.);
  }
`;

export function segmentIntegral(emission, extinction, distance) {
  const tau=extinction*distance;
  return emission*distance*(Math.abs(tau)<1e-6?1-tau/2+tau*tau/6:-Math.expm1(-tau)/tau);
}

const halton=(index,base)=>{let value=0,fraction=1;while(index>0){fraction/=base;value+=fraction*(index%base);index=Math.floor(index/base);}return value;};
const depthOffsets=[.5,.125,.875,.375,.625,.0625,.9375,.5];

export class VolumePreview {
  constructor(meta, buffers) {
    this.meta=meta;this.targets=new Map();this.steps=192;this.dustScale=1;this.showStars=true;this.showYoung=true;
    this.textures=buffers.map(buffer=>{
      if(buffer.byteLength!==meta.size**3*4*4)throw new Error('Volume length does not match metadata');
      const texture=new THREE.Data3DTexture(new Float32Array(buffer),meta.size,meta.size,meta.size);
      texture.type=THREE.FloatType;texture.format=THREE.RGBAFormat;
      texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.unpackAlignment=1;texture.needsUpdate=true;
      return texture;
    });
    this.material=new THREE.ShaderMaterial({glslVersion:THREE.GLSL3,vertexShader:volumeVertexShader,fragmentShader:volumeFragmentShader,
      depthTest:false,depthWrite:false,toneMapped:false,uniforms:{
        starsDust:{value:this.textures[0]},youngSources:{value:this.textures[1]},history:{value:null},
        inverseProjection:{value:new THREE.Matrix4()},cameraWorld:{value:new THREE.Matrix4()},
        boxMin:{value:new THREE.Vector3(...meta.bounds_kpc[0])},boxMax:{value:new THREE.Vector3(...meta.bounds_kpc[1])},
        extinction:{value:new THREE.Vector3(...meta.extinction_per_kpc)},resolution:{value:new THREE.Vector2()},
        pixelJitter:{value:new THREE.Vector2()},depthJitter:{value:.5},dustScale:{value:1},
        showStars:{value:1},showYoung:{value:1},steps:{value:192},sampleIndex:{value:0},
      }});
    this.scene=new THREE.Scene();this.quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material);this.scene.add(this.quad);
    this.camera=new THREE.Camera();
  }
  reset(){for(const target of this.targets.values())target.samples=0;}
  get needsFrame(){return !this.targets.size||[...this.targets.values()].some(target=>target.samples<8);}
  render(renderer,camera,width,height){
    if(!renderer.extensions.has('OES_texture_float_linear')||!renderer.extensions.has('EXT_color_buffer_float'))throw new Error('This GPU needs floating-point textures and render targets; use the diagnostic view.');
    let state=this.targets.get(renderer);
    if(!state){
      const make=()=>new THREE.WebGLRenderTarget(width,height,{type:THREE.FloatType,depthBuffer:false,stencilBuffer:false});
      state={buffers:[make(),make()],samples:0};this.targets.set(renderer,state);
      for(const buffer of state.buffers){renderer.setRenderTarget(buffer);renderer.setClearColor(0,0);renderer.clear();}
    }
    if(state.buffers[0].width!==width||state.buffers[0].height!==height){
      state.buffers.forEach(buffer=>buffer.setSize(width,height));state.samples=0;
    }
    if(state.samples>=8)return state.buffers[(state.samples-1)%2].texture;
    camera.updateMatrixWorld();
    const uniforms=this.material.uniforms,index=state.samples;
    uniforms.inverseProjection.value.copy(camera.projectionMatrixInverse);uniforms.cameraWorld.value.copy(camera.matrixWorld);
    uniforms.resolution.value.set(width,height);uniforms.sampleIndex.value=index;
    uniforms.pixelJitter.value.set(index?halton(index,2)-.5:0,index?halton(index,3)-.5:0);
    uniforms.depthJitter.value=depthOffsets[index];
    uniforms.steps.value=this.steps;uniforms.dustScale.value=this.dustScale;
    uniforms.showStars.value=Number(this.showStars);uniforms.showYoung.value=Number(this.showYoung);
    uniforms.history.value=state.buffers[(index+1)%2].texture;
    renderer.setRenderTarget(state.buffers[index%2]);renderer.render(this.scene,this.camera);
    state.samples++;
    renderer.domElement.dataset.volumeSamples=String(state.samples);
    return state.buffers[index%2].texture;
  }
  dispose(){
    this.textures.forEach(texture=>texture.dispose());this.material.dispose();this.quad.geometry.dispose();
    for(const state of this.targets.values())state.buffers.forEach(buffer=>buffer.dispose());
    this.targets.clear();
  }
}
