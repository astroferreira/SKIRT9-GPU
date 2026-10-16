import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { cameraBasis, wrapAngle, MAX_FOV_KPC, MAX_TARGET_KPC } from './camera.js';
import { VolumePreview } from './volume.js';
import { BackendVolumePreview, backendDimensions } from './backend-preview.js';

const layerStyle = {
  stars: {gain:1.0, minSize:2.0, classId:0},
  young: {gain:1.25, minSize:2.8, classId:1},
  gas: {gain:1.35, minSize:3.0, classId:2},
  dust: {gain:.24, minSize:2.5, classId:3},
};

export const previewShaderDefaults = Object.freeze({
  stellarKernel:1.0,
  diffuseKernel:1.5,
  psfSigma:0,
  structure:0,
  stretch:0,
});

const vertexShader = `
  attribute vec3 sourcePosition;
  attribute vec3 color;
  attribute float kernelRadius;
  attribute float strength;
  varying vec3 vColor;
  varying float vPeak;
  varying vec2 vUv;
  uniform vec2 viewportSize;
  uniform float viewportHeight;
  uniform float minPointSize;
  uniform float sampleWeight;
  uniform float kernelScale;

  void main() {
    vColor = color;
    vUv = uv;
    vec4 viewPosition = modelViewMatrix * vec4(sourcePosition, 1.0);
    gl_Position = projectionMatrix * viewPosition;
    float projectedDiameter = kernelRadius * kernelScale * projectionMatrix[1][1] * viewportHeight;
    if (projectionMatrix[3][3] == 0.0) projectedDiameter /= max(-viewPosition.z, 1e-4);
    float diameter = max(minPointSize, projectedDiameter);
    // Instanced billboards have no hardware point-size cap. Pixel filtering
    // broadens unresolved kernels while the inverse-area factor retains light.
    gl_Position.xy += 2.0*position.xy*diameter/viewportSize*gl_Position.w;
    if (projectionMatrix[3][3] == 0.0 && viewPosition.z >= 0.0) gl_Position=vec4(2.,2.,2.,1.);
    // Preserve integrated light as the physical kernel changes on screen.
    // sampleWeight makes a uniform display sample an unbiased estimate of the
    // complete input catalogue used by SKIRT.
    vPeak = strength * sampleWeight * 24.0 / max(diameter * diameter, 1.0);
  }
`;

const fragmentShader = `
  varying vec2 vUv;
  varying vec3 vColor;
  varying float vPeak;
  uniform float layerGain;
  uniform float layerClass;
  uniform float cubicKernel;

  float projectedCubicSpline(float q) {
    if (q >= 1.0) return 0.0;
    float q2 = q*q;
    if (q < 1e-4) return 1.0 - 8.0*log(2.0)*q2;
    float s = sqrt(max(0.0, (1.0-q)*(1.0+q)));
    float sigma;
    if (q < 0.5) {
      float t = sqrt(max(0.0, (1.0-2.0*q)*(1.0+2.0*q)));
      float p1 = (4.0+26.0*q2)*s;
      float p2 = (1.0+26.0*q2)*t;
      float p3 = 18.0*q2*q2*log(max(2.0*q/(1.0+t), 1e-7));
      float p4 = 6.0*q2*(4.0+q2)*log(max(2.0*(1.0+s)/(1.0+t), 1e-7));
      sigma = (2.0/3.14159265359)*(p1-p2-p3-p4);
    } else {
      float p1 = (2.0+13.0*q2)*s;
      float p2 = 3.0*q2*(4.0+q2)*log(max(q/(1.0+s), 1e-7));
      sigma = (4.0/3.14159265359)*(p1+p2);
    }
    return max(0.0, sigma * 3.14159265359 / 6.0);
  }

  void main() {
    float q = length(vUv-0.5)*2.0;
    if (q >= 1.0) discard;
    float soft = exp(-3.5*q*q) * (1.0-smoothstep(.82, 1.0, q));
    float profile = mix(soft, projectedCubicSpline(q), cubicKernel);
    float core = exp(-22.0*q*q);
    bool stellar = layerClass < 1.5;
    vec3 displayColor = stellar ? mix(vColor, vec3(1.0), .68*core) : vColor;
    // A compact core plus the physical halo creates the luminous appearance
    // of the reference galaxy shader without replacing the SKIRT kernel.
    if (stellar) profile = .72*profile + .28*core;
    profile *= 1.0-smoothstep(1.0-max(fwidth(q),.001),1.0,q);
    float radiance = profile*vPeak*layerGain;
    if (radiance < .00002) discard;
    // Pre-weight radiance so the floating-point target adds source light
    // linearly even when an individual compact source is brighter than one.
    gl_FragColor = vec4(displayColor*radiance, 1.0);
  }
`;

const compositeVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const compositeFragmentShader = `
  precision highp float;
  uniform sampler2D radianceTexture;
  uniform sampler2D blurTexture;
  uniform float exposure;
  uniform float psfMix;
  uniform float structureStrength;
  uniform int stretchMode;
  varying vec2 vUv;

  vec3 asinhColor(vec3 value) {
    return log(value + sqrt(value*value + 1.0));
  }

  vec3 acesFilmic(vec3 value) {
    return clamp((value*(2.51*value+.03))/(value*(2.43*value+.59)+.14), 0.0, 1.0);
  }

  void main() {
    vec3 sharp = max(texture2D(radianceTexture, vUv).rgb*exposure, vec3(0.0));
    vec3 blurred = max(texture2D(blurTexture, vUv).rgb*exposure, vec3(0.0));
    if (stretchMode == 1) {
      sharp = log(sharp + 1.0);
      blurred = log(blurred + 1.0);
    } else if (stretchMode == 2) {
      sharp = asinhColor(sharp*10.0)/10.0;
      blurred = asinhColor(blurred*10.0)/10.0;
    }
    vec3 color = mix(sharp, blurred, psfMix);
    color = max(color + (sharp-blurred)*structureStrength, vec3(0.0));
    color = acesFilmic(color);
    // Bright stellar piles become neutral-white, as broadband SKIRT products
    // do, while low surface-brightness age and gas colors remain legible.
    float highlight = smoothstep(.72, 1.0, max(max(color.r, color.g), color.b));
    color = mix(color, vec3(dot(color, vec3(.2126,.7152,.0722))), .12*highlight);
    gl_FragColor = vec4(color, 1.0);
    #include <colorspace_fragment>
  }
`;

const blurFragmentShader = `
  precision highp float;
  uniform sampler2D radianceTexture;
  uniform vec2 resolution;
  uniform vec2 direction;
  uniform float sigma;
  varying vec2 vUv;

  void main() {
    vec2 off1 = 1.411764705882353*direction*sigma/resolution;
    vec2 off2 = 3.294117647058823*direction*sigma/resolution;
    vec2 off3 = 5.176470588235294*direction*sigma/resolution;
    vec3 color = min(texture2D(radianceTexture,vUv).rgb,vec3(100.0))*.1964825501511404;
    color += min(texture2D(radianceTexture,vUv+off1).rgb,vec3(100.0))*.2969069646728344;
    color += min(texture2D(radianceTexture,vUv-off1).rgb,vec3(100.0))*.2969069646728344;
    color += min(texture2D(radianceTexture,vUv+off2).rgb,vec3(100.0))*.09447039785044732;
    color += min(texture2D(radianceTexture,vUv-off2).rgb,vec3(100.0))*.09447039785044732;
    color += min(texture2D(radianceTexture,vUv+off3).rgb,vec3(100.0))*.010381362401148057;
    color += min(texture2D(radianceTexture,vUv-off3).rgb,vec3(100.0))*.010381362401148057;
    gl_FragColor = vec4(color,1.0);
  }
`;

// Map the compact UI range onto photographic stops. This retains useful
// control for both sparse catalogues and very dense galactic nuclei.
export const displayExposure = value => .002*Math.pow(100,value);

export class GalaxyViewer {
  constructor(container, miniContainer, onCameraChange, onPickComplete) {
    this.container = container;
    this.onCameraChange = onCameraChange;
    this.onPickComplete = onPickComplete;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#000000');
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
    this.miniRenderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.miniRenderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.miniRenderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.append(this.renderer.domElement);
    miniContainer.append(this.miniRenderer.domElement);
    this.sceneCamera = new THREE.PerspectiveCamera(45, 1, .01, 10000);
    this.sceneCamera.up.set(0,0,1);
    this.sceneCamera.layers.enable(1);
    this.observerCamera = new THREE.OrthographicCamera(-30,30,30,-30,.01,10000);
    this.miniCamera = new THREE.OrthographicCamera(-30,30,30,-30,.01,10000);
    this.controls = new OrbitControls(this.sceneCamera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.addEventListener('change', () => this.dirty = true);
    this.particles = new THREE.Group();
    this.pickParticles = new THREE.Group();
    this.scene.add(this.particles);
    this.guide = new THREE.Group();
    this.scene.add(this.guide);
    this.targetMarker = new THREE.Group();
    this.scene.add(this.targetMarker);
    this.layers = {stars:true, young:true, gas:true, dust:false};
    this.mode = 'scene';
    this.extent = 60;
    this.dirty = true;
    this.gridVisible = true;
    this.brightness = .38;
    this.shaderControls = {...previewShaderDefaults};
    this.previewMode='diagnostic';this.volume=null;
    this.volumeControls={steps:192,dust:true,dustScale:1,renderScale:.75};
    this.radianceTargets = new Map();
    this.compositeMaterial = new THREE.ShaderMaterial({
      uniforms:{radianceTexture:{value:null},blurTexture:{value:null},exposure:{value:displayExposure(this.brightness)},
                psfMix:{value:1},structureStrength:{value:0},stretchMode:{value:0}},
      vertexShader:compositeVertexShader,fragmentShader:compositeFragmentShader,
      depthTest:false,depthWrite:false,toneMapped:false,
    });
    this.compositeScene = new THREE.Scene();
    this.compositeScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.compositeMaterial));
    this.compositeCamera = new THREE.Camera();
    this.blurMaterial = new THREE.ShaderMaterial({
      uniforms:{radianceTexture:{value:null},resolution:{value:new THREE.Vector2(1,1)},
                direction:{value:new THREE.Vector2(1,0)},sigma:{value:this.shaderControls.psfSigma}},
      vertexShader:compositeVertexShader,fragmentShader:blurFragmentShader,
      depthTest:false,depthWrite:false,toneMapped:false,blending:THREE.NoBlending,
    });
    this.blurScene = new THREE.Scene();
    this.blurScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.blurMaterial));
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resizeObserver.observe(miniContainer);
    this.miniContainer = miniContainer;
    this.installInteraction();
    const draw = () => {
      this.animation = requestAnimationFrame(draw);
      if (!this.camera || (!this.dirty && !(['volume','backend'].includes(this.previewMode) && (this.volume?.needsFrame||this.volume?.hasUpdated)))) return;
      if(this.dirty)this.volume?.reset();
      try {
        if(this.previewMode==='backend'&&this.volume?.remote){
          this.volume.update(this.backendRequest(),(error,meta)=>{
            if(error){this.setPreviewMode('diagnostic');this.container.dispatchEvent(new CustomEvent('previewerror',{detail:`Backend GPU: ${error.message}. Browser mode remains available.`}));}
            else this.container.dispatchEvent(new CustomEvent('backendframe',{detail:meta}));
          });
        }
        this.renderView(this.renderer, this.mode === 'scene' ? this.sceneCamera : this.observerCamera);
        this.renderView(this.miniRenderer, this.miniCamera);
        if(this.volume?.remote)this.volume.hasUpdated=false;
      } catch(error) {
        this.setPreviewMode('diagnostic');
        this.container.dispatchEvent(new CustomEvent('previewerror',{detail:error.message}));
        this.renderView(this.renderer,this.mode==='scene'?this.sceneCamera:this.observerCamera);
      }
      this.dirty = false;
    };
    draw();
  }

  clearGroup(group) {
    while (group.children.length) {
      const child = group.children[0];
      child.geometry?.dispose();
      child.material?.dispose();
      group.remove(child);
    }
  }

  renderView(renderer, camera) {
    if(this.previewMode==='backend'&&this.volume?.remote){
      this.volume.render(renderer,renderer===this.renderer?'main':'mini');
      const mask=camera.layers.mask,background=this.scene.background;
      if((mask&2)!==0){
        camera.layers.set(1);this.scene.background=null;renderer.autoClear=false;renderer.clearDepth();
        renderer.render(this.scene,camera);renderer.autoClear=true;camera.layers.mask=mask;this.scene.background=background;
      }
      return;
    }
    const drawingBuffer = renderer.getDrawingBufferSize(new THREE.Vector2());
    if(this.previewMode==='volume' && this.volume)drawingBuffer.multiplyScalar(this.volumeControls.renderScale).floor();
    for (const points of this.particles.children) {
      points.material.uniforms.viewportHeight.value = drawingBuffer.y;
      points.material.uniforms.viewportSize.value.copy(drawingBuffer);
    }
    let targets = this.radianceTargets.get(renderer);
    if (!targets) {
      const makeTarget = () => new THREE.WebGLRenderTarget(drawingBuffer.x,drawingBuffer.y,{
        type:THREE.HalfFloatType,format:THREE.RGBAFormat,
        minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,
        depthBuffer:false,stencilBuffer:false,
      });
      targets={radiance:makeTarget(),blurX:makeTarget(),blurY:makeTarget()};
      targets.radiance.texture.name='Galaxy radiance';
      targets.blurY.texture.name='Galaxy PSF';
      this.radianceTargets.set(renderer,targets);
    } else if (targets.radiance.width!==drawingBuffer.x || targets.radiance.height!==drawingBuffer.y) {
      Object.values(targets).forEach(target=>target.setSize(drawingBuffer.x,drawingBuffer.y));
    }

    const cameraMask=camera.layers.mask, background=this.scene.background;
    camera.layers.set(0);
    this.scene.background=null;
    renderer.setRenderTarget(targets.radiance);
    renderer.setClearColor(0x000000,0);
    renderer.clear(true,true,true);
    let radianceTexture=targets.radiance.texture;
    if(this.previewMode==='volume' && this.volume){
      this.volume.steps=this.volumeControls.steps;
      this.volume.dustScale=this.volumeControls.dust?this.volumeControls.dustScale:0;
      this.volume.showStars=this.layers.stars;this.volume.showYoung=this.layers.young;
      radianceTexture=this.volume.render(renderer,camera,drawingBuffer.x,drawingBuffer.y);
    } else renderer.render(this.scene,camera);

    const needsBlur=this.shaderControls.psfSigma>0 || this.shaderControls.structure>0;
    if (needsBlur) {
      this.blurMaterial.uniforms.resolution.value.copy(drawingBuffer);
      this.blurMaterial.uniforms.sigma.value=this.shaderControls.psfSigma>0?this.shaderControls.psfSigma:1;
      this.blurMaterial.uniforms.radianceTexture.value=radianceTexture;
      this.blurMaterial.uniforms.direction.value.set(1,0);
      renderer.setRenderTarget(targets.blurX);
      renderer.clear(true,true,true);
      renderer.render(this.blurScene,this.compositeCamera);
      this.blurMaterial.uniforms.radianceTexture.value=targets.blurX.texture;
      this.blurMaterial.uniforms.direction.value.set(0,1);
      renderer.setRenderTarget(targets.blurY);
      renderer.clear(true,true,true);
      renderer.render(this.blurScene,this.compositeCamera);
    }

    renderer.setRenderTarget(null);
    renderer.setClearColor(0x000000,1);
    renderer.clear(true,true,true);
    this.compositeMaterial.uniforms.radianceTexture.value=radianceTexture;
    this.compositeMaterial.uniforms.blurTexture.value=needsBlur?targets.blurY.texture:radianceTexture;
    this.compositeMaterial.uniforms.exposure.value=displayExposure(this.brightness);
    this.compositeMaterial.uniforms.psfMix.value=this.shaderControls.psfSigma>0?1:0;
    this.compositeMaterial.uniforms.structureStrength.value=this.shaderControls.structure;
    this.compositeMaterial.uniforms.stretchMode.value=this.shaderControls.stretch;
    renderer.render(this.compositeScene,this.compositeCamera);

    // The framing guides stay crisp and are excluded from HDR accumulation.
    if ((cameraMask & 2)!==0) {
      camera.layers.set(1);
      renderer.autoClear=false;
      renderer.clearDepth();
      renderer.render(this.scene,camera);
      renderer.autoClear=true;
    }
    camera.layers.mask=cameraMask;
    this.scene.background=background;
  }

  async load(meta, signal) {
    const buffers = await Promise.all(meta.layers.map(async layer => {
      const response = await fetch(layer.url, {signal});
      if (!response.ok) throw new Error(`Could not read ${layer.kind} catalogue (${response.status})`);
      return {layer, values: new Float32Array(await response.arrayBuffer())};
    }));
    if (signal.aborted) return;
    this.clearGroup(this.particles);
    this.clearGroup(this.pickParticles);
    this.clearVolume();
    let bound = 0;
    for (const {layer, values} of buffers) {
      if (meta.stride_floats !== 8 || values.length !== layer.display_count * meta.stride_floats) throw new Error('Catalogue binary length does not match metadata');
      const geometry = new THREE.InstancedBufferGeometry();
      const quad=new THREE.PlaneGeometry(1,1);geometry.index=quad.index;geometry.attributes.position=quad.attributes.position;geometry.attributes.uv=quad.attributes.uv;
      geometry.instanceCount=layer.display_count;
      const interleaved = new THREE.InstancedInterleavedBuffer(values, meta.stride_floats);
      geometry.setAttribute('sourcePosition', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));
      geometry.setAttribute('color', new THREE.InterleavedBufferAttribute(interleaved, 3, 3));
      geometry.setAttribute('kernelRadius', new THREE.InterleavedBufferAttribute(interleaved, 1, 6));
      geometry.setAttribute('strength', new THREE.InterleavedBufferAttribute(interleaved, 1, 7));
      const pickGeometry=new THREE.BufferGeometry();
      pickGeometry.setAttribute('position',new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(values,meta.stride_floats),3,0));
      const pickPoints=new THREE.Points(pickGeometry,new THREE.PointsMaterial());pickPoints.userData.kind=layer.kind;this.pickParticles.add(pickPoints);
      const style = layerStyle[layer.kind] || layerStyle.stars;
      const kernelScale=['stars','young'].includes(layer.kind)?this.shaderControls.stellarKernel:this.shaderControls.diffuseKernel;
      const material = new THREE.ShaderMaterial({
        uniforms: {viewportHeight:{value:1},viewportSize:{value:new THREE.Vector2(1,1)}, minPointSize:{value:style.minSize},
                   cubicKernel:{value:layer.kernel?.profile?.startsWith('SKIRT')?1:0},
                   layerClass:{value:style.classId}, layerGain:{value:style.gain},
                   kernelScale:{value:kernelScale},
                   sampleWeight:{value:Math.max(1,(layer.positive_count||layer.count)/layer.display_count)}},
        vertexShader, fragmentShader,
        transparent: true, blending:THREE.AdditiveBlending, depthWrite:false,
      });
      const points = new THREE.Mesh(geometry, material);points.frustumCulled=false;
      points.userData.kind = layer.kind;
      points.visible = this.layers[layer.kind];
      this.particles.add(points);
      bound = Math.max(bound, ...layer.bounds_kpc.flat().map(Math.abs));
    }
    this.extent = Math.max(10, bound * 2);
    if (this.grid) {this.grid.geometry.dispose();this.grid.material.dispose();this.scene.remove(this.grid);}
    const gridSize = Math.max(20, Math.ceil(this.extent / 10) * 10);
    this.grid = new THREE.GridHelper(gridSize, 20, '#344e4b', '#192c2e');
    this.grid.rotation.x = Math.PI / 2;
    this.grid.layers.set(1);
    this.grid.visible = this.gridVisible;
    this.grid.material.transparent = true;
    this.grid.material.opacity = .42;
    this.scene.add(this.grid);
    this.resetScene();
    this.updateCamera(this.camera);
  }

  resetScene() {
    const span = Math.max(this.camera?.fov_kpc || 60, this.extent * .7);
    this.sceneCamera.position.set(span*1.1,-span*1.2,span*.9);
    this.sceneCamera.near = .01;
    this.sceneCamera.far = Math.max(10000,this.extent*30);
    this.sceneCamera.updateProjectionMatrix();
    this.controls.target.set(0,0,0);
    this.controls.update();
    this.dirty = true;
  }

  setLayer(kind, visible) {
    this.layers[kind] = visible;
    this.particles.children.filter(x => x.userData.kind === kind).forEach(x => x.visible = visible);
    this.dirty = true;
  }

  clearVolume(){this.volume?.dispose();this.volume=null;this.dirty=true;}

  backendRequest(){
    const interactive=this.volume.interactive;
    const views=[[this.renderer,this.mode==='scene'?this.sceneCamera:this.observerCamera,'main'],[this.miniRenderer,this.miniCamera,'mini']].map(([renderer,camera,name])=>{
      camera.updateMatrixWorld();
      const size=renderer.getDrawingBufferSize(new THREE.Vector2());
      return {name,...backendDimensions(size.x,size.y,this.volumeControls.renderScale*(interactive?.5:1)),
        inverse_projection:camera.projectionMatrixInverse.toArray(),camera_world:camera.matrixWorld.toArray()};
    });
    return {views,controls:{steps:interactive?96:this.volumeControls.steps,dust:this.volumeControls.dust,
      dust_scale:this.volumeControls.dustScale,stars:this.layers.stars,young:this.layers.young,
      exposure:displayExposure(this.brightness),psf:this.shaderControls.psfSigma,structure:this.shaderControls.structure,
      stretch:this.shaderControls.stretch}};
  }

  async loadVolume(meta,signal,backend=false,accessKey=''){
    if(backend){
      const volume=await BackendVolumePreview.create(meta,signal,accessKey);
      if(signal.aborted){volume.dispose();return;}
      this.clearVolume();this.volume=volume;this.dirty=true;return;
    }
    for(const renderer of [this.renderer,this.miniRenderer]){
      const gl=renderer.getContext(),limit=gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
      if(meta.size>limit)throw new Error(`This browser supports at most ${limit}³ volume textures. Choose a smaller grid.`);
    }
    const buffers=await Promise.all(['stars-dust.bin','young.bin'].map(async name=>{
      const response=await fetch(meta.files[name],{signal});
      if(!response.ok)throw new Error(`Could not read preview volume (${response.status})`);
      return response.arrayBuffer();
    }));
    if(signal.aborted)return;
    const volume=new VolumePreview(meta,buffers);this.clearVolume();this.volume=volume;this.dirty=true;
  }

  setPreviewMode(mode){
    this.previewMode=['volume','backend'].includes(mode)?mode:'diagnostic';
    this.renderer.domElement.dataset.previewMode=this.previewMode;this.dirty=true;
  }

  setVolumeControl(name,value){
    if(!(name in this.volumeControls))return;
    this.volumeControls[name]=value;this.dirty=true;
  }

  setBrightness(value) {
    this.brightness = value;
    this.compositeMaterial.uniforms.exposure.value=displayExposure(value);
    this.dirty = true;
  }

  setShaderControl(name,value) {
    if (!(name in previewShaderDefaults)) return;
    this.shaderControls[name]=value;
    if (name==='stellarKernel' || name==='diffuseKernel') {
      const kinds=name==='stellarKernel'?['stars','young']:['gas','dust'];
      this.particles.children.filter(points=>kinds.includes(points.userData.kind))
        .forEach(points=>points.material.uniforms.kernelScale.value=value);
    }
    this.dirty=true;
  }

  resetShaderControls() {
    for (const [name,value] of Object.entries(previewShaderDefaults)) this.setShaderControl(name,value);
    return {...this.shaderControls};
  }

  setPickMode(enabled) {
    this.pickMode=enabled;
    this.renderer.domElement.classList.toggle('picking-target',enabled);
  }

  pickTarget(event) {
    const rect=this.renderer.domElement.getBoundingClientRect();
    const pointer=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    const activeCamera=this.mode==='scene'?this.sceneCamera:this.observerCamera;
    const distance=this.mode==='scene'?activeCamera.position.distanceTo(this.controls.target):Math.max(this.extent*3,1000);
    const worldPerPixel=this.mode==='scene'?2*distance*Math.tan(THREE.MathUtils.degToRad(activeCamera.fov/2))/rect.height:this.camera.fov_kpc/this.frameSide;
    const raycaster=new THREE.Raycaster();raycaster.params.Points.threshold=Math.max(worldPerPixel*7,.05);raycaster.setFromCamera(pointer,activeCamera);
    const hits=raycaster.intersectObjects(this.pickParticles.children.filter(p=>this.layers[p.userData.kind]),false);
    if(!hits.length){this.onPickComplete?.(null);return;}
    const near=hits.filter(hit=>(hit.distanceToRay??0)<=raycaster.params.Points.threshold*.7);
    const points=(near.length?near:hits).map(hit=>hit.point);
    const coordinate=axis=>{const values=points.map(p=>p.getComponent(axis)).sort((a,b)=>a-b);const middle=Math.floor(values.length/2);return values.length%2?values[middle]:(values[middle-1]+values[middle])/2;};
    const target=new THREE.Vector3(coordinate(0),coordinate(1),coordinate(2));
    target.set(...target.toArray().map(v=>THREE.MathUtils.clamp(v,-MAX_TARGET_KPC,MAX_TARGET_KPC)));
    this.clearGroup(this.targetMarker);
    const outer=Math.max(.5,this.camera.fov_kpc*.024),inner=Math.max(.35,outer*.72);
    const marker=new THREE.Mesh(new THREE.RingGeometry(inner,outer,32),new THREE.MeshBasicMaterial({color:'#c9f4ce',side:THREE.DoubleSide,depthTest:false,transparent:true,opacity:.9}));
    marker.position.copy(target);marker.quaternion.copy(activeCamera.quaternion);marker.layers.set(1);this.targetMarker.add(marker);
    if(this.mode==='scene'){this.controls.target.copy(target);this.controls.update();}
    this.onCameraChange({...this.camera,target_kpc:target.toArray()});this.onPickComplete?.(target.toArray());this.dirty=true;
  }

  setMode(mode) {
    this.mode = mode;
    this.controls.enabled = mode === 'scene';
    this.resize();
  }

  setSEDMode(enabled) {
    this.sedMode=enabled;
    this.guide.visible=!enabled;
    this.updateCamera(this.camera);
  }

  updateCamera(camera) {
    if (!camera) return;
    this.camera = structuredClone(camera);
    const basis = cameraBasis(camera);
    const target = new THREE.Vector3(...(this.sedMode?[0,0,0]:camera.target_kpc));
    const direction = new THREE.Vector3(...basis.observer);
    const up = new THREE.Vector3(...basis.up);
    const distance = Math.max(this.extent * 3, 1000);
    for (const c of [this.observerCamera, this.miniCamera]) {
      c.position.copy(target).addScaledVector(direction, distance);
      c.up.copy(up);
      c.near = .01;
      c.far = distance * 2 + this.extent * 3;
      c.lookAt(target);
      c.updateMatrixWorld();
    }
    this.clearGroup(this.guide);
    const right = new THREE.Vector3(...basis.right);
    const half = camera.fov_kpc / 2;
    const depth = Math.max(camera.fov_kpc * .35, this.extent * .25);
    const corners = [-depth,0,depth].map(d => [[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y]) =>
      target.clone().addScaledVector(right,x*half).addScaledVector(up,y*half).addScaledVector(direction,d)));
    const segments = [];
    for (const plane of corners) for(let i=0;i<4;i++) segments.push(plane[i],plane[(i+1)%4]);
    for(let i=0;i<4;i++) segments.push(corners[0][i],corners[2][i]);
    const wire = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segments),
      new THREE.LineBasicMaterial({color:'#8abfa0',transparent:true,opacity:.37,depthTest:false}));
    wire.layers.set(1);
    this.guide.add(wire);
    const cameraPoint = target.clone().addScaledVector(direction, depth*1.2);
    const glyph = new THREE.Mesh(new THREE.OctahedronGeometry(Math.max(.4,camera.fov_kpc*.025)),
      new THREE.MeshBasicMaterial({color:'#a1d7ad',wireframe:true,depthTest:false}));
    glyph.position.copy(cameraPoint);glyph.layers.set(1);this.guide.add(glyph);
    const aim = new THREE.Line(new THREE.BufferGeometry().setFromPoints([target,cameraPoint]),
      new THREE.LineBasicMaterial({color:'#73987b',transparent:true,opacity:.5,depthTest:false}));
    aim.layers.set(1);this.guide.add(aim);
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w,h,false);
    this.sceneCamera.aspect = w/h;this.sceneCamera.updateProjectionMatrix();
    this.miniRenderer.setSize(this.miniContainer.clientWidth || 176,this.miniContainer.clientWidth || 176,false);
    if (this.camera) {
      const side = Math.max(100,Math.min(w-52,h-170));
      const half = (this.sedMode?Math.max(this.extent*1.1,this.camera.fov_kpc):this.camera.fov_kpc)/2;
      Object.assign(this.observerCamera,{left:-half*w/side,right:half*w/side,top:half*h/side,bottom:-half*h/side});
      Object.assign(this.miniCamera,{left:-half,right:half,top:half,bottom:-half});
      this.observerCamera.updateProjectionMatrix();this.miniCamera.updateProjectionMatrix();
      const guide = document.getElementById('frame-guide');
      guide.style.width = guide.style.height = `${side}px`;
      this.frameSide = side;
    }
    this.dirty = true;
  }

  captureCamera() {
    const direction = this.sceneCamera.position.clone().sub(this.controls.target).normalize();
    const inclination = Math.acos(THREE.MathUtils.clamp(direction.z,-1,1))*180/Math.PI;
    const azimuth = Math.atan2(direction.y,direction.x)*180/Math.PI;
    const zero = cameraBasis({inclination,azimuth,roll:0});
    const actualUp = new THREE.Vector3(0,1,0).applyQuaternion(this.sceneCamera.quaternion);
    const roll = Math.atan2(actualUp.dot(new THREE.Vector3(...zero.right)),actualUp.dot(new THREE.Vector3(...zero.up)))*180/Math.PI;
    this.onCameraChange({...this.camera,inclination,azimuth,roll:wrapAngle(roll),target_kpc:this.controls.target.toArray()});
  }

  installInteraction() {
    const canvas = this.renderer.domElement;
    let drag = null;
    canvas.addEventListener('pointerdown',e=>{
      if(!this.pickMode||e.button!==0)return;
      e.preventDefault();e.stopImmediatePropagation();this.setPickMode(false);this.pickTarget(e);
    },true);
    canvas.addEventListener('contextmenu',e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      if (this.mode !== 'camera') return;
      drag = {id:e.pointerId,x:e.clientX,y:e.clientY,pan:e.button===2 || e.shiftKey};
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove',e => {
      if(!drag || drag.id!==e.pointerId || this.mode!=='camera') return;
      const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
      drag.x=e.clientX;drag.y=e.clientY;
      const camera=structuredClone(this.camera);
      if(drag.pan){
        if(this.sedMode)return;
        const {right,up}=cameraBasis(camera),scale=camera.fov_kpc/this.frameSide;
        camera.target_kpc=camera.target_kpc.map((v,i)=>THREE.MathUtils.clamp(v-dx*scale*right[i]+dy*scale*up[i],-MAX_TARGET_KPC,MAX_TARGET_KPC));
      }else{
        camera.azimuth=wrapAngle(camera.azimuth-dx*.35);
        camera.inclination=THREE.MathUtils.clamp(camera.inclination+dy*.35,0,180);
      }
      this.onCameraChange(camera);
    });
    const end=e=>{if(drag?.id===e.pointerId)drag=null;};
    canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);canvas.addEventListener('lostpointercapture',end);
    canvas.addEventListener('wheel',e=>{
      if(this.mode!=='camera'||this.sedMode)return;
      e.preventDefault();
      this.onCameraChange({...this.camera,fov_kpc:THREE.MathUtils.clamp(this.camera.fov_kpc*Math.exp(e.deltaY*.001),.1,MAX_FOV_KPC)});
    },{passive:false});
  }
}
