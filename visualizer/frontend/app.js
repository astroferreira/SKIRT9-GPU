import { GalaxyViewer, previewShaderDefaults } from './viewer.js';
import { defaults, preset, MAX_FOV_KPC, MAX_TARGET_KPC } from './camera.js';
import { SpectrumPlot } from './spectrum.js';
import { volumeGridInfo } from './volume.js';

const $ = id => document.getElementById(id);
let config, dataset, viewer, camera=defaults(60), cloudController, selectedJob=null, jobs=[], pollInFlight=false, cloudReady=false;
let spectrumPlot, spectrumLoadedId=null, spectrumLoadingId=null, seed=12345, selectedProduct='image', displayedJobId=null;
let volumeController=null,volumeStatus='',backendStatus=null;
const activeStates = ['queued','preparing','running','processing'];
const productOrder = ['image','rgb','sed'];
const productNames = {image:'Broadband',rgb:'RGB',sed:'SED'};
const formatCount = x => new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:2}).format(x);
const duration = x => x<60 ? `${Math.floor(x)}s` : `${Math.floor(x/60)}m ${Math.floor(x%60)}s`;
const hide = (id, value=true) => $(id).hidden=value;
function error(message) { $('error-banner').textContent=message; hide('error-banner',!message); }
async function api(url,options={}) {
  const response=await fetch(url,{...options,headers:{'Content-Type':'application/json',...options.headers}});
  if(!response.ok){let body;try{body=await response.json();}catch{body={detail:response.statusText};}throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));}
  return response.json();
}
const instrumentsFor=request=>request?.instruments?.length?[...request.instruments]:[request?.output||'image'];
const hasProduct=(request,product)=>instrumentsFor(request).includes(product);
const selectedInstruments=()=>productOrder.filter(product=>$(`instrument-${product}`).checked);
const sedSelected=()=>$('instrument-sed').checked;
const spatialSelected=()=>$('instrument-image').checked||$('instrument-rgb').checked;
const sedOnlySelected=()=>sedSelected()&&!spatialSelected();
const rgbBands=()=>({red:$('rgb-red').value,green:$('rgb-green').value,blue:$('rgb-blue').value});
const rgbValid=()=>new Set(Object.values(rgbBands())).size===3;
const shaderSliders={
  stellarKernel:{id:'shader-stellar-kernel',format:value=>`${value.toFixed(1)}×`},
  diffuseKernel:{id:'shader-diffuse-kernel',format:value=>`${value.toFixed(1)}×`},
  psfSigma:{id:'shader-psf',format:value=>`${Number.isInteger(value)?value:value.toFixed(2)} px`},
  structure:{id:'shader-structure',format:value=>`${value.toFixed(1)}×`},
};
function setShaderSlider(name,value){
  const spec=shaderSliders[name],input=$(spec.id);input.value=String(value);
  $(`${spec.id}-value`).textContent=spec.format(Number(value));viewer?.setShaderControl(name,Number(value));
}
function resetShaderUI(){
  const values=viewer?.resetShaderControls()||previewShaderDefaults;
  for(const [name,value] of Object.entries(values)){
    if(name==='stretch'){$('shader-stretch').value=String(value);viewer?.setShaderControl(name,value);}
    else setShaderSlider(name,value);
  }
}

function updatePreviewMode(){
  const mode=$('preview-mode').value,volume=['volume','backend'].includes(mode),backend=mode==='backend';
  hide('volume-controls',!volume);
  hide('backend-access',!backend||backendStatus?.auth_required===false);
  hide('backend-frame-status',!backend);
  updateVolumeGridHint();
  document.querySelector('.stellar-age-key').hidden=volume;
  document.querySelector('[data-layer="dust"]').closest('label').hidden=volume;
  hide('gas-layer-label',volume||!dataset?.raw);
  hide('young-layer-label',Boolean(dataset?.raw)&&!volume);
  if(viewer.volume&&(Boolean(viewer.volume.remote)!==backend||(!volume&&viewer.volume.remote)))viewer.clearVolume();
  viewer.setPreviewMode(volume&&viewer.volume?mode:'diagnostic');
  if(volume&&cloudReady&&!viewer.volume)buildVolume();
  if(!volume){volumeController?.abort();volumeController=null;$('preview-rebuild').disabled=false;}
}

function updateVolumeGridHint(){
  const size=Number($('preview-resolution').value),span=Number($('preview-span').value);
  if(!$('preview-span').checkValidity()){
    $('preview-grid-hint').textContent='Enter a valid cube width to estimate voxel size.';
    return;
  }
  const {voxelPc,textureMiB}=volumeGridInfo(size,span);
  if($('preview-mode').value==='backend'){
    $('preview-grid-hint').textContent=`${voxelPc.toLocaleString('en',{maximumFractionDigits:1})} pc per voxel · ${textureMiB.toLocaleString('en')} MiB of shared server GPU textures, plus working memory. Only image frames reach this browser. Build / update applies grid changes; zoom does not refine the grid.`;
    return;
  }
  const warning=size>=512?'High memory: 512³ needs 4 GiB of volume data and about 8 GiB of GPU textures, plus working memory. ':'';
  $('preview-grid-hint').textContent=`${warning}${voxelPc.toLocaleString('en',{maximumFractionDigits:1})} pc per voxel · ${textureMiB.toLocaleString('en')} MiB download · ~${(2*textureMiB).toLocaleString('en')} MiB GPU textures for both views, plus browser and render-buffer memory. Higher grids take longer to prepare. Zooming does not refine the grid; rebuild after changing these settings.`;
}

async function buildVolume(){
  if(!cloudReady||!dataset)return;
  if(!$('preview-span').reportValidity()||!$('preview-kernel').reportValidity())return;
  const mode=$('preview-mode').value,backend=mode==='backend';
  if(backend&&backendStatus?.auth_required!==false&&!$('backend-access-key').value.trim()){
    $('preview-status').dataset.state='error';$('preview-status').textContent='Enter the backend GPU access key, then click Build / update.';return;
  }
  volumeController?.abort();const controller=new AbortController();volumeController=controller;
  const identifier=dataset.id;
  const query=new URLSearchParams({red:$('preview-red').value,green:$('preview-green').value,blue:$('preview-blue').value,
    size:$('preview-resolution').value,span:$('preview-span').value,kernel:$('preview-kernel').value,
    x:camera.target_kpc[0],y:camera.target_kpc[1],z:camera.target_kpc[2]});
  $('preview-rebuild').disabled=true;
  $('preview-status').dataset.state='preparing';
  $('preview-status').textContent='Preparing full-catalogue band emission and dust volume… First preparation can take a few minutes; subsequent views are cached.';
  try{
    const meta=await api(`/api/datasets/${identifier}/preview-volume?${query}`,{signal:controller.signal});
    await viewer.loadVolume(meta,controller.signal,backend,$('backend-access-key').value.trim());
    if(controller.signal.aborted||dataset.id!==identifier)return;
    if($('preview-mode').value===mode)viewer.setPreviewMode(mode);
    const bands=meta.bands.map(b=>config.bands[b].label).join(' / ');
    volumeStatus=`${bands} · ${meta.size}³ · ${meta.voxel_kpc.toPrecision(3)} kpc voxels · ${formatCount(Object.values(meta.counts).reduce((a,b)=>a+b,0))} input sites.`;
    $('preview-status').title=`${meta.redshift>0?`Observer z=${meta.redshift.toFixed(2)}`:'Rest frame'}. Cube centred at ${[meta.request.x,meta.request.y,meta.request.z].map(v=>Number(v).toFixed(1)).join(', ')} kpc.`;
    $('preview-status').textContent=volumeStatus;
    $('preview-limits').textContent=meta.warnings.join(' ');
    $('preview-status').dataset.state='ready';
  }catch(e){
    if(e.name!=='AbortError'){
      $('preview-status').textContent=`${viewer.volume?'Previous volume retained.':'Diagnostic view retained.'} ${e.message}`;
      $('preview-status').dataset.state='error';
    }
  }finally{if(volumeController===controller){$('preview-rebuild').disabled=false;volumeController=null;}}
}
const resolutionFor=request=>request?.spectral_resolution??config.spectral_modes[request.spectral_mode]?.resolving_power;
function productLabel(request,product){
  if(product==='sed'){const mode=config.spectral_modes?.[request.spectral_mode];return `${mode?.group.replace('JWST · ','')||'SED'} · ${mode?.label||request.spectral_mode} · R ${resolutionFor(request)?.toLocaleString()}`;}
  if(product==='rgb'){const bands=request.rgb_bands||{};return `RGB · ${['red','green','blue'].map(c=>config.bands[bands[c]]?.label||bands[c]||'?').join(' / ')}`;}
  return config.bands[request.band]?.label||request.band;
}
function observationLabel(request){const selected=instrumentsFor(request);return selected.length===1?productLabel(request,selected[0]):selected.map(product=>productNames[product]).join(' + ');}
function currentRequest(){const instruments=selectedInstruments();const output=instruments.includes('rgb')?'rgb':instruments[0]||'image';return{dataset_id:dataset.id,band:$('band').value,rgb_bands:rgbBands(),camera:structuredClone(camera),quality:$('quality').value,pixels:Number($('pixels').value),seed,output,instruments,spectral_mode:$('spectral-mode').value,spectral_resolution:sedSelected()?Number($('spectral-resolution').value):null,model_fov_kpc:dataset.raw?Number($('model-fov').value):null,source_velocities:$('source-velocities').checked};}
function updateSubmitState(){
  const instruments=selectedInstruments();
  const modelValid=!dataset?.raw||($('model-fov').checkValidity()&&(!spatialSelected()||Number($('model-fov').value)>=camera.fov_kpc));
  $('render').disabled=!cloudReady||!config.backend.available||dataset?.render_available===false||!instruments.length||(sedSelected()&&!$('spectral-resolution').checkValidity())||($('instrument-rgb').checked&&!rgbValid())||!modelValid;
}
function sameRequest(a,b){
  if(!a||!b)return false;
  const ai=instrumentsFor(a),bi=instrumentsFor(b);if(ai.length!==bi.length||ai.some((value,index)=>value!==bi[index]))return false;
  if(a.dataset_id!==b.dataset_id||a.quality!==b.quality||a.seed!==b.seed)return false;
  if((a.model_fov_kpc??null)!==(b.model_fov_kpc??null))return false;
  if(Boolean(a.source_velocities)!==Boolean(b.source_velocities))return false;
  if(hasProduct(a,'sed')&&(a.spectral_mode!==b.spectral_mode||resolutionFor(a)!==resolutionFor(b)))return false;
  if(hasProduct(a,'image')&&a.band!==b.band)return false;
  if(hasProduct(a,'rgb')&&['red','green','blue'].some(c=>(a.rgb_bands||{})[c]!==b.rgb_bands[c]))return false;
  const spatial=hasProduct(a,'image')||hasProduct(a,'rgb');
  if(!spatial)return ['inclination','azimuth'].every(k=>Math.abs(a.camera[k]-b.camera[k])<1e-7);
  return a.pixels===b.pixels&&['inclination','azimuth','roll','fov_kpc'].every(k=>Math.abs(a.camera[k]-b.camera[k])<1e-7)&&a.camera.target_kpc.every((v,i)=>Math.abs(v-b.camera.target_kpc[i])<1e-7);
}
function staleCheck(){hide('stale-notice',!selectedJob||selectedJob.status!=='completed'||sameRequest(selectedJob.request,currentRequest()));}
function updateCamera(next){
  camera=structuredClone(next);
  for(const key of ['inclination','azimuth','roll']){$(key).value=camera[key];$(`${key}-value`).textContent=`${camera[key].toFixed(0)}°`;}
  $('fov').max=Math.max(200,camera.fov_kpc);$('fov').value=camera.fov_kpc;$('fov-value').value=Number(camera.fov_kpc.toFixed(2));
  ['x','y','z'].forEach((axis,i)=>$(`target-${axis}`).value=Number(camera.target_kpc[i].toFixed(3)));
  $('preview-fov').textContent=`${camera.fov_kpc.toFixed(1)} × ${camera.fov_kpc.toFixed(1)} kpc`;
  document.querySelectorAll('[data-preset]').forEach(button=>{
    const p=preset(button.dataset.preset,camera.fov_kpc);
    button.classList.toggle('selected',['inclination','azimuth','roll'].every(k=>Math.abs(camera[k]-p[k])<.01));
  });
  viewer?.updateCamera(camera);if(dataset){staleCheck();updateSubmitState();}
}
function setMode(mode){
  viewer.setMode(mode);$('scene-mode').classList.toggle('active',mode==='scene');$('camera-mode').classList.toggle('active',mode==='camera');
  hide('capture-camera',mode==='camera');hide('preview-inset',mode==='camera'||sedOnlySelected());hide('frame-guide',mode!=='camera'||sedOnlySelected());
  hide('scene-stats',mode==='camera');hide('toggle-grid',mode==='camera');
  $('navigation-hint').textContent=mode==='camera'?'Orbit camera':'Orbit scene';$('zoom-hint').textContent=mode==='camera'?(sedOnlySelected()?'Whole model':'Change FOV'):'Zoom scene';
}
function setPickMode(enabled){viewer.setPickMode(enabled);$('pick-target').classList.toggle('active',enabled);$('pick-target').textContent=enabled?'Click the companion…':'⌖ Pick center';$('pick-target').setAttribute('aria-pressed',String(enabled));}
function updateObservationMode(){
  const sed=sedSelected(),spatial=spatialSelected(),instruments=selectedInstruments();document.body.dataset.observation=sed&&!spatial?'sed':'image';
  hide('band-settings',!$('instrument-image').checked);hide('rgb-settings',!$('instrument-rgb').checked);hide('spectral-settings',!sed);hide('pixel-settings',!spatial);
  for(const id of ['fov','fov-value','target-x','target-y','target-z','roll'])$(id).disabled=!spatial;
  $('camera-note').textContent=!spatial&&sed?'SED: whole galaxy · inclination and azimuth only':sed?'Shared direction · SED ignores roll, target, and FOV':'Distant observer · square physical FOV';
  $('render-label').textContent=dataset?.render_available===false?'Full halo · view only':!instruments.length?'Select an instrument':instruments.length===1?`Run ${productNames[instruments[0]]} with SKIRTGPU`:`Run ${instruments.length} instruments with SKIRTGPU`;
  $('camera-mode').textContent=!spatial&&sed?'Viewing direction':'Camera view';
  if(viewer){viewer.setSEDMode(!spatial&&sed);setMode(viewer.mode);}
  const mode=config.spectral_modes[$('spectral-mode').value];
  const r=Number($('spectral-resolution').value),valid=$('spectral-resolution').checkValidity();
  $('spectral-range').textContent=`${mode.min_micron}–${mode.max_micron} μm · ${valid?`ideal R ≈ ${r.toLocaleString()}`:'enter a valid R'} · ${dataset?.distance_mpc>0?'rest frame':'observer frame'}`;
  $('resolution-hint').textContent=valid?`R = λ/Δλ · ${(1+Math.ceil(Math.log(mode.max_micron/mode.min_micron)/Math.log1p(1/(r*mode.samples_per_resolution_element)))).toLocaleString()} samples · 2 samples per resolution element`:`Enter a whole-number R from ${$('spectral-resolution').min} to ${$('spectral-resolution').max}.`;
  if(dataset){const scope=dataset.raw?`On-demand ${Number($('model-fov').value).toLocaleString()} kpc model`:(!spatial&&sed?'SED observer only':'Full catalogue');$('render-note').textContent=dataset.render_available===false?dataset.render_unavailable_reason:`${scope} · ${dataset.dust_emission.toLowerCase()} dust`;staleCheck();}
  updateSubmitState();
}
function updateBand(){const band=config.bands[$('band').value];$('band-pivot').textContent=`λ ≈ ${band.pivot_micron.toFixed(3)} μm`;
  $('band-frame').textContent=dataset?.distance_mpc>0?'Rest frame at fixed distance':'Observer frame';if(dataset)staleCheck();}
async function loadDataset(identifier, restoredCamera=null){
  volumeController?.abort();volumeController=null;volumeStatus='';
  viewer?.clearVolume();viewer?.setPreviewMode('diagnostic');
  cloudController?.abort();const controller=new AbortController();cloudController=controller;
  if(viewer)setPickMode(false);
  dataset=config.datasets.find(d=>d.id===identifier);$('dataset').value=identifier;cloudReady=false;$('render').disabled=true;
  $('preview-span').value=dataset.fov_kpc;$('preview-status').dataset.state='idle';$('preview-rebuild').disabled=false;
  updateVolumeGridHint();
  hide('model-region-settings',!dataset.raw);hide('gas-layer-label',!dataset.raw);hide('young-layer-label',Boolean(dataset.raw));$('dust-layer-name').textContent=dataset.raw?'Dust proxy':'Dust';
  if(dataset.raw)$('model-fov').value=dataset.default_model_fov_kpc;
  $('source-velocities').disabled=!dataset.source_velocities?.available||dataset.render_available===false;
  $('source-velocities').closest('label').title=dataset.source_velocities?.reason||"Resolved stellar and star-forming gas velocities relative to the model's mass-weighted stellar velocity; no unresolved dispersion.";
  if($('source-velocities').disabled)$('source-velocities').checked=false;
  hide('cloud-loading',false);$('cloud-loading').querySelector('strong').textContent='Opening the catalogue';
  $('dataset-description').textContent=dataset.description;$('scene-name').textContent=dataset.name;
  $('redshift').textContent=`z = ${dataset.observer_redshift.toFixed(2)}`;$('particle-count').textContent='…';
  $('model-note').textContent=`${dataset.physics ? `BPASS/TODDLERS · snapshot z = ${dataset.physics.snapshot_redshift.toFixed(3)} · ` : ''}${dataset.dust_emission} dust · ${dataset.distance_mpc>0?`${dataset.distance_mpc} Mpc observer`:`observer z = ${dataset.observer_redshift}`} · No PSF or detector noise`;
  $('render-note').textContent=`${dataset.raw?`On-demand ${dataset.default_model_fov_kpc.toLocaleString()} kpc model`:'Full catalogue'} · ${dataset.dust_emission.toLowerCase()} dust`;
  updateCamera(restoredCamera||defaults(Math.min(MAX_FOV_KPC,Math.max(.1,dataset.fov_kpc))));updateBand();updateObservationMode();error('');
  try{
    const meta=await api(`/api/datasets/${identifier}/cloud`,{signal:controller.signal});
    await viewer.load(meta,controller.signal);if(controller.signal.aborted)return;
    $('particle-count').textContent=formatCount(meta.layers.filter(x=>dataset.raw?x.kind==='stars':x.kind!=='dust').reduce((s,l)=>s+l.count,0));
    const display=meta.layers.reduce((s,l)=>s+l.display_count,0),total=meta.layers.reduce((s,l)=>s+l.count,0);
    $('scene-stats').textContent=dataset.raw?`${formatCount(display)} sampled sites / ${formatCount(total)} raw particles · full-count HDR weighting · 32-neighbour stellar kernels · stars by age · gas in purple.`:`${formatCount(display)} sampled sites / ${formatCount(total)} catalogue entries · full-count HDR weighting · 32-neighbour stellar kernels · stars by age.`;
    $('scene-stats').title=`${meta.sampling}\n${meta.coloring}\n${dataset.notes||''}`;
    cloudReady=true;hide('cloud-loading');updateSubmitState();updatePreviewMode();
  }catch(e){if(e.name!=='AbortError'){error(e.message);$('cloud-loading').querySelector('strong').textContent='Catalogue unavailable';}}
}
function populateHistory(){
  $('history-count').textContent=jobs.length;
  if(!jobs.length)return;
  const fragment=document.createDocumentFragment();
  for(const job of jobs){
    const card=document.createElement('button');card.className=`history-card${job.id===selectedJob?.id?' active':''}`;card.dataset.jobId=job.id;
    if(job.preview_url||job.image_url){const image=document.createElement('img');image.src=job.preview_url||job.image_url;image.alt=`${job.dataset_name}, ${observationLabel(job.request)}`;image.loading='lazy';card.append(image);}
    else{const placeholder=document.createElement('div');placeholder.className='history-placeholder';placeholder.textContent=activeStates.includes(job.status)?'◌':'◎';card.append(placeholder);}
    const details=document.createElement('div');const title=document.createElement('strong');const onlySed=instrumentsFor(job.request).length===1&&hasProduct(job.request,'sed');title.textContent=onlySed?observationLabel(job.request):`${observationLabel(job.request)} · ${job.request.camera.fov_kpc.toFixed(1)} kpc`;title.title=title.textContent;
    const info=document.createElement('p');info.textContent=`i ${job.request.camera.inclination.toFixed(0)}° / φ ${job.request.camera.azimuth.toFixed(0)}° · ${onlySed?'integrated SED':`${job.request.pixels} px`}`;
    const state=document.createElement('p');state.textContent=`${job.status} · ${duration(job.elapsed_seconds)}`;
    const name=document.createElement('small');name.textContent=job.dataset_name;details.append(title,info,state,name);card.append(details);
    card.addEventListener('click',()=>{selectedJob=job;showJob();populateHistory();});fragment.append(card);
  }
  $('history').replaceChildren(fragment);
}
async function loadSpectrum(job){
  if(spectrumLoadedId===job.id){hide('spectrum-section',false);return;}
  if(spectrumLoadingId===job.id)return;
  spectrumLoadingId=job.id;hide('spectrum-section');
  try{
    const data=await api(job.products?.sed?.spectrum_url||job.spectrum_url);
    if(selectedJob?.id!==job.id||selectedProduct!=='sed')return;
    spectrumPlot.setData(data);spectrumLoadedId=job.id;
    $('spectrum-title').textContent=`${job.dataset_name} · ${productLabel(job.request,'sed')}`;
    $('spectrum-description').textContent=`Whole model · ${data.samples.toLocaleString()} spectral bins · ${data.mode.min_micron}–${data.mode.max_micron} μm · ideal R ≈ ${data.mode.resolving_power.toLocaleString()}`;
    $('spectrum-readout').textContent='Hover to inspect · drag horizontally to zoom';hide('spectrum-section',false);
  }catch(e){if(selectedJob?.id===job.id)error(`Could not load spectrum: ${e.message}`);}
  finally{if(spectrumLoadingId===job.id)spectrumLoadingId=null;}
}
function showJob(){
  const job=selectedJob;if(!job)return;
  const available=instrumentsFor(job.request),active=activeStates.includes(job.status),complete=job.status==='completed';
  if(displayedJobId!==job.id){selectedProduct=job.primary_product||job.request.output||available[0];displayedJobId=job.id;}
  if(!available.includes(selectedProduct))selectedProduct=available[0];
  for(const button of $('product-tabs').querySelectorAll('button')){const shown=available.includes(button.dataset.product);button.hidden=!shown;button.classList.toggle('active',shown&&button.dataset.product===selectedProduct);}
  hide('product-tabs',!complete||available.length<2);
  const selectedIsSed=selectedProduct==='sed';
  $('result-state').textContent=job.status.toUpperCase();hide('render-loading',!active);hide('result-empty',active||complete);hide('rendered-image',!complete);
  hide('artifact-links',!complete);hide('execution-details',job.status==='queued');
  hide('download-csv',!complete||!selectedIsSed);hide('download-sed',!complete||!selectedIsSed);hide('explore-spectrum',!complete||!selectedIsSed);
  if(!complete||!selectedIsSed)hide('spectrum-section');
  if(active){$('job-phase').textContent=job.phase;$('job-timer').textContent=`${duration(job.elapsed_seconds)} elapsed${job.phase_percent!==null?` · ${job.phase_percent.toFixed(1)}% of phase`:''}`;
    $('job-progress').classList.toggle('determinate',job.phase_percent!==null);$('job-progress').style.width=job.phase_percent!==null?`${job.phase_percent}%`:'25%';
    $('cancel').disabled=job.cancel_requested;$('cancel').textContent=job.cancel_requested?'Cancelling…':'Cancel observation';}
  if(complete){
    const product=job.products?.[selectedProduct]||job,preview=product.preview_url||product.image_url;
    if($('rendered-image').getAttribute('src')!==preview)$('rendered-image').src=preview;
    $('rendered-image').classList.toggle('spectrum-thumbnail',selectedIsSed);
    $('rendered-image').alt=`${job.dataset_name}, ${productLabel(job.request,selectedProduct)}, inclination ${job.request.camera.inclination} degrees${selectedIsSed?', integrated spectrum':`, ${job.request.camera.fov_kpc} kpc FOV`}`;
    $('download-fits').href=product.fits_url;$('download-fits').textContent=selectedProduct==='rgb'?'↓ FITS cube':'↓ FITS';$('download-png').href=preview;$('download-png').download=selectedIsSed?'spectrum.png':selectedProduct==='rgb'?'rgb.png':'observation.png';$('download-manifest').href=`/api/jobs/${job.id}/files/manifest.json`;
    if(selectedIsSed){
      $('download-csv').href=product.csv_url;$('download-sed').href=product.raw_url;
      $('result-details').textContent=`${productLabel(job.request,'sed')} · ${job.spectrum?.samples} bins · λFλ (W/m²) · ${duration(job.elapsed_seconds)}${job.gpu_confirmed?' · GPU confirmed':''}`;
      loadSpectrum(job);
    }else{const metadata=selectedProduct==='rgb'?job.rgb:job.image;$('result-details').textContent=`${productLabel(job.request,selectedProduct)} · ${job.request.pixels} × ${job.request.pixels} px · ${duration(job.elapsed_seconds)} · ${metadata?.units||''}${job.gpu_confirmed?' · GPU confirmed':''}`;}
  }else if(active){$('result-details').textContent=`${job.dataset_name} · ${observationLabel(job.request)} · You can keep composing while this observation runs.`;}
  else{$('result-details').textContent=job.error||`Observation ${job.status}. You can submit another render.`;}
  $('render-warnings').textContent=(job.warnings||[]).join('\n');staleCheck();
  $('execution-details').querySelector('summary').textContent=job.warnings?.length?'Model warnings & render log':'Render details & log';
}
async function poll(){
  if(pollInFlight||!config)return;pollInFlight=true;
  try{
    jobs=(await api('/api/jobs')).jobs;
    if(selectedJob)selectedJob=jobs.find(j=>j.id===selectedJob.id)||selectedJob;
    else if(jobs.length)selectedJob=jobs[0];
    showJob();populateHistory();
    $('backend-status').textContent=config.backend.available?`${jobs.filter(j=>activeStates.includes(j.status)).length||'No'} active observations`:'Renderer unavailable';
    if(selectedJob&&$('execution-details').open){const log=await api(`/api/jobs/${selectedJob.id}/log`);$('job-log').textContent=log.text;}
  }catch(e){$('backend-status').textContent='Reconnecting to local server…';}finally{pollInFlight=false;}
}
async function start(){
  try{
    config=await api('/api/config');
    $('backend-status').textContent=config.backend.available?'Renderer connected':'Renderer unavailable';$('backend-dot').classList.toggle('off',!config.backend.available);
    const staticGroup=document.createElement('optgroup'),mooseGroup=document.createElement('optgroup');staticGroup.label='Prepared models';mooseGroup.label='MOOSE raw snapshots';
    for(const d of config.datasets){const option=new Option(d.name,d.id);option.disabled=!d.available;(d.raw?mooseGroup:staticGroup).append(option);}
    $('dataset').replaceChildren(staticGroup,mooseGroup);$('dataset').disabled=false;
    const fillBands=(element,value)=>{const groups=new Map();for(const [id,band] of Object.entries(config.bands)){if(!groups.has(band.group)){const group=document.createElement('optgroup');group.label=band.group;groups.set(band.group,group);}groups.get(band.group).append(new Option(band.label,id));}element.replaceChildren(...groups.values());element.value=value;element.disabled=false;};
    fillBands($('band'),'JWST_NIRCAM_F150W');fillBands($('rgb-red'),'JWST_NIRCAM_F444W');fillBands($('rgb-green'),'JWST_NIRCAM_F200W');fillBands($('rgb-blue'),'JWST_NIRCAM_F115W');
    fillBands($('preview-red'),'SLOAN_SDSS_I');fillBands($('preview-green'),'SLOAN_SDSS_R');fillBands($('preview-blue'),'SLOAN_SDSS_G');
    const spectralGroups=new Map();for(const [id,mode] of Object.entries(config.spectral_modes)){if(!spectralGroups.has(mode.group)){const group=document.createElement('optgroup');group.label=mode.group;spectralGroups.set(mode.group,group);}spectralGroups.get(mode.group).append(new Option(mode.label,id));}
    $('spectral-mode').replaceChildren(...spectralGroups.values());$('spectral-mode').value='nirspec_prism';$('spectral-mode').disabled=false;
    $('spectral-resolution').min=config.spectral_resolution?.min??10;$('spectral-resolution').max=config.spectral_resolution?.max??10000;
    $('spectral-resolution').value=config.spectral_modes.nirspec_prism.resolving_power;
    $('fov-value').max=MAX_FOV_KPC;['x','y','z'].forEach(axis=>{$(`target-${axis}`).min=-MAX_TARGET_KPC;$(`target-${axis}`).max=MAX_TARGET_KPC;});
    spectrumPlot=new SpectrumPlot($('spectrum-chart'),$('spectrum-readout'),{select:$('spectrum-feature-select'),details:$('spectrum-feature-details'),status:$('spectrum-feature-status')});
    document.querySelectorAll('[data-feature-group]').forEach(input=>input.addEventListener('change',()=>spectrumPlot.setFeatureGroup(input.dataset.featureGroup,input.checked)));
    viewer=new GalaxyViewer($('viewport'),$('mini-viewport'),updateCamera,target=>{setPickMode(false);if(!target)error('No displayed particle was found there. Try clicking the brighter center of the companion.');else error('');});updateCamera(camera);
    backendStatus=await api('/api/backend-preview/status').catch(()=>null);
    $('preview-mode').querySelector('[value="backend"]').disabled=!backendStatus?.available;
    $('preview-mode').addEventListener('change',updatePreviewMode);
    $('preview-rebuild').addEventListener('click',buildVolume);
    const pendingVolume=()=>{$('preview-status').textContent=`${volumeStatus} Settings changed: click Build / update to apply.`;$('preview-status').dataset.state='pending';};
    for(const id of ['preview-red','preview-green','preview-blue','preview-span','preview-resolution','preview-kernel'])$(id).addEventListener('change',pendingVolume);
    $('preview-resolution').addEventListener('change',updateVolumeGridHint);
    $('preview-span').addEventListener('input',updateVolumeGridHint);
    $('preview-use-rgb').addEventListener('click',()=>{for(const channel of ['red','green','blue'])$(`preview-${channel}`).value=$(`rgb-${channel}`).value;pendingVolume();});
    $('preview-steps').addEventListener('change',()=>viewer.setVolumeControl('steps',Number($('preview-steps').value)));
    $('preview-scale').addEventListener('change',()=>viewer.setVolumeControl('renderScale',Number($('preview-scale').value)));
    $('preview-dust').addEventListener('change',()=>viewer.setVolumeControl('dust',$('preview-dust').checked));
    $('preview-dust-scale').addEventListener('input',()=>{const value=Number($('preview-dust-scale').value);viewer.setVolumeControl('dustScale',value);$('preview-dust-value').textContent=`${value.toFixed(2)}×`;});
    $('viewport').addEventListener('previewerror',event=>{$('preview-mode').value='diagnostic';updatePreviewMode();error(event.detail);});
    $('viewport').addEventListener('backendframe',event=>{
      const meta=event.detail;
      $('backend-frame-status').textContent=`${meta.gpu.renderer} · GPU ${meta.gpu.device} · ${meta.views[0].samples}/8 samples · ${meta.render_encode_ms.toFixed(0)} ms render + PNG encoding (network excluded)`;
    });
    $('dataset').addEventListener('change',()=>loadDataset($('dataset').value));
    document.querySelectorAll('[data-instrument]').forEach(input=>input.addEventListener('change',updateObservationMode));
    const resetResolution=()=>{$('spectral-resolution').value=config.spectral_modes[$('spectral-mode').value].resolving_power;updateObservationMode();};
    $('spectral-mode').addEventListener('change',resetResolution);$('reset-resolution').addEventListener('click',resetResolution);
    $('spectral-resolution').addEventListener('input',updateObservationMode);
    $('model-fov').addEventListener('input',()=>{updateObservationMode();staleCheck();updateSubmitState();});
    $('spectrum-raw').addEventListener('change',()=>spectrumPlot.setRaw($('spectrum-raw').checked));
    $('spectrum-scale').addEventListener('click',()=>{spectrumPlot.setLog(!spectrumPlot.logFlux);$('spectrum-scale').textContent=spectrumPlot.logFlux?'Log flux':'Linear flux';$('spectrum-scale').setAttribute('aria-pressed',String(spectrumPlot.logFlux));});
    $('spectrum-reset').addEventListener('click',()=>spectrumPlot.reset());
    $('explore-spectrum').addEventListener('click',()=>$('spectrum-section').scrollIntoView({behavior:'smooth',block:'center'}));
    $('band').addEventListener('change',updateBand);for(const id of ['rgb-red','rgb-green','rgb-blue'])$(id).addEventListener('change',()=>{staleCheck();updateSubmitState();});['quality','pixels','source-velocities'].forEach(id=>$(id).addEventListener('change',staleCheck));
    for(const key of ['inclination','azimuth','roll'])$(key).addEventListener('input',()=>updateCamera({...camera,[key]:Number($(key).value)}));
    $('fov').addEventListener('input',()=>updateCamera({...camera,fov_kpc:Number($('fov').value)}));
    $('fov-value').addEventListener('change',()=>{if($('fov-value').checkValidity())updateCamera({...camera,fov_kpc:Number($('fov-value').value)});else updateCamera(camera);});
    ['x','y','z'].forEach((axis,i)=>$(`target-${axis}`).addEventListener('change',()=>{const input=$(`target-${axis}`);if(input.checkValidity()&&input.value!==''){const target=[...camera.target_kpc];target[i]=Number(input.value);updateCamera({...camera,target_kpc:target});}else updateCamera(camera);}));
    document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>updateCamera(preset(button.dataset.preset,camera.fov_kpc))));
    document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>viewer.setLayer(input.dataset.layer,input.checked)));
    $('brightness').addEventListener('input',()=>viewer.setBrightness(Number($('brightness').value)));
    for(const [name,spec] of Object.entries(shaderSliders))$(spec.id).addEventListener('input',()=>setShaderSlider(name,Number($(spec.id).value)));
    $('shader-stretch').addEventListener('change',()=>viewer.setShaderControl('stretch',Number($('shader-stretch').value)));
    $('reset-shader').addEventListener('click',resetShaderUI);
    resetShaderUI();
    $('reset-camera').addEventListener('click',()=>{updateCamera(defaults(dataset.fov_kpc));viewer.resetScene();});
    $('scene-mode').addEventListener('click',()=>setMode('scene'));$('camera-mode').addEventListener('click',()=>setMode('camera'));$('capture-camera').addEventListener('click',()=>viewer.captureCamera());$('pick-target').addEventListener('click',()=>setPickMode(!viewer.pickMode));
    $('toggle-grid').addEventListener('click',()=>{viewer.gridVisible=!viewer.gridVisible;if(viewer.grid)viewer.grid.visible=viewer.gridVisible;viewer.dirty=true;$('toggle-grid').textContent=viewer.gridVisible?'Grid on':'Grid off';$('toggle-grid').setAttribute('aria-pressed',String(viewer.gridVisible));});
    $('product-tabs').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{selectedProduct=button.dataset.product;showJob();}));
    $('render').addEventListener('click',async()=>{
      if(dataset.render_available===false||(sedSelected()&&!$('spectral-resolution').reportValidity()))return;
      if($('instrument-rgb').checked&&!rgbValid()){error('Choose three distinct filters for the RGB channels.');return;}
      $('render').disabled=true;error('');
      try{const result=await api('/api/jobs',{method:'POST',body:JSON.stringify(currentRequest())});selectedJob=result.job;showJob();await poll();}
      catch(e){error(e.message);}finally{updateSubmitState();}
    });
    $('cancel').addEventListener('click',async()=>{if(!selectedJob)return;try{selectedJob=await api(`/api/jobs/${selectedJob.id}/cancel`,{method:'POST'});showJob();await poll();}catch(e){error(e.message);}});
    $('restore-camera').addEventListener('click',async()=>{if(!selectedJob)return;const request=structuredClone(selectedJob.request);$('band').value=request.band;$('quality').value=request.quality;$('pixels').value=request.pixels;seed=request.seed;
      const restored=instrumentsFor(request);for(const product of productOrder)$(`instrument-${product}`).checked=restored.includes(product);const channels=request.rgb_bands||{red:'JWST_NIRCAM_F444W',green:'JWST_NIRCAM_F200W',blue:'JWST_NIRCAM_F115W'};for(const color of ['red','green','blue'])$(`rgb-${color}`).value=channels[color];$('spectral-mode').value=request.spectral_mode||'nirspec_prism';
      $('spectral-resolution').value=request.spectral_resolution??config.spectral_modes[$('spectral-mode').value].resolving_power;
      $('source-velocities').checked=Boolean(request.source_velocities);
      if(request.model_fov_kpc)$('model-fov').value=request.model_fov_kpc;
      if(request.dataset_id!==dataset.id)await loadDataset(request.dataset_id,request.camera);else updateCamera(request.camera);updateBand();updateObservationMode();setMode('camera');});
    $('execution-details').addEventListener('toggle',()=>{if($('execution-details').open)poll();});
    const available=config.datasets.find(d=>d.available&&d.id===config.default_dataset)||config.datasets.find(d=>d.available);
    if(!available)throw new Error('No available galaxy catalogues. Check datasets.json.');
    await loadDataset(available.id);await poll();setInterval(poll,2000);
  }catch(e){error(`Could not open the observatory: ${e.message}`);hide('cloud-loading');$('backend-status').textContent='Setup needs attention';}
}
start();
