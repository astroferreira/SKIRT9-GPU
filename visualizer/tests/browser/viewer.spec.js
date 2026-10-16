import { test, expect } from '@playwright/test';

test('Catalogue, camera, FOV, layer controls and responsive layout',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  await page.goto('/');await expect(page.locator('#cloud-loading')).toBeHidden({timeout:120000});
  await expect(page.locator('#error-banner')).toBeHidden();
  await expect(page.locator('#render')).toBeEnabled();
  await expect(page.locator('#particle-count')).not.toHaveText('—');
  await expect(page.locator('#quality option')).toHaveCount(5);
  await expect(page.locator('#quality option').nth(3)).toHaveText('Ultra · 1B');
  await expect(page.locator('#quality option').nth(4)).toHaveText('Extreme · 10B');
  await expect(page.locator('.stellar-age-key')).toBeVisible();
  await expect(page.locator('#scene-stats')).toContainText('32-neighbour stellar kernels');
  await page.locator('.shader-controls summary').click();
  await expect(page.locator('#shader-stellar-kernel')).toHaveValue('1');
  await expect(page.locator('#shader-diffuse-kernel')).toHaveValue('1.5');
  await page.locator('#shader-stellar-kernel').fill('3.2');await expect(page.locator('#shader-stellar-kernel-value')).toHaveText('3.2×');
  await page.locator('#shader-psf').fill('1.25');await expect(page.locator('#shader-psf-value')).toHaveText('1.25 px');
  await page.locator('#shader-structure').fill('0.6');await page.locator('#shader-stretch').selectOption('2');
  await page.locator('#reset-shader').click();await expect(page.locator('#shader-stellar-kernel')).toHaveValue('1');
  await expect(page.locator('#shader-psf')).toHaveValue('0');await expect(page.locator('#shader-stretch')).toHaveValue('0');
  await page.locator('.shader-controls summary').click();
  await page.locator('#camera-mode').click();await expect(page.locator('#frame-guide')).toBeVisible();
  await expect(page.locator('#camera-mode')).toHaveClass('active');
  await page.locator('[data-preset="xy"]').click();await expect(page.locator('#inclination-value')).toHaveText('0°');
  await page.locator('#fov-value').fill('24');await page.locator('#fov-value').press('Tab');
  await expect(page.locator('#preview-fov')).toHaveText('24.0 × 24.0 kpc');
  const box=await page.locator('#viewport').boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+60,box.y+box.height/2+30,{steps:5});await page.mouse.up();
  await expect(page.locator('#inclination-value')).not.toHaveText('0°');
  await page.mouse.down({button:'right'});await page.mouse.move(box.x+box.width/2+90,box.y+box.height/2+50,{steps:3});await page.mouse.up({button:'right'});
  expect(await page.locator('#target-x').inputValue()).not.toBe('0');
  await page.locator('[data-layer="dust"]').check();await page.locator('[data-layer="dust"]').uncheck();
  await page.locator('#reset-camera').click();await page.locator('#scene-mode').click();
  await page.screenshot({path:'test-results/observatory-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.locator('.shader-controls summary').click();
  await page.screenshot({path:'test-results/observatory-mobile.png',fullPage:true});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('Instrument checkboxes submit broadband, RGB channels, and SED in one run',async({page})=>{
  let submitted,job=null;
  await page.route('**/api/mock-multi-spectrum',route=>route.fulfill({json:{
    wavelength_micron:[.7,2,5.3],raw_flux:[1,2,1],smoothed_flux:[1,2,1],samples:3,
    mode:{label:'Prism',min_micron:.7,max_micron:5.3,resolving_power:100},spectral_frame:'observer'}}));
  await page.route('**/api/jobs',async route=>{
    if(route.request().method()==='POST'){
      submitted=route.request().postDataJSON();const pixel='data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
      job={id:'multi-browser',status:'completed',dataset_name:'Test',elapsed_seconds:1,phase:'Observation ready',warnings:[],request:submitted,primary_product:'rgb',
        image:{units:'W/m2/arcsec2'},rgb:{units:'W/m2/arcsec2'},spectrum:{samples:3},products:{
          image:{preview_url:pixel,fits_url:'data:application/octet-stream;base64,AA=='},
          rgb:{preview_url:pixel,fits_url:'data:application/octet-stream;base64,AA=='},
          sed:{preview_url:pixel,fits_url:'data:application/octet-stream;base64,AA==',spectrum_url:'/api/mock-multi-spectrum',csv_url:'#csv',raw_url:'#raw'}}};
      await route.fulfill({status:202,json:{job,reused:false}});
    }else await route.fulfill({json:{jobs:job?[job]:[]}});
  });
  await page.goto('/');await expect(page.locator('#render')).toBeEnabled({timeout:120000});
  await page.locator('#instrument-rgb').check();await page.locator('#instrument-sed').check();
  await expect(page.locator('#band-settings')).toBeVisible();await expect(page.locator('#rgb-settings')).toBeVisible();await expect(page.locator('#spectral-settings')).toBeVisible();
  await expect(page.locator('#fov')).toBeEnabled();await expect(page.locator('#render-label')).toHaveText('Run 3 instruments with SKIRTGPU');
  await page.locator('#rgb-red').selectOption('JWST_NIRCAM_F115W');await expect(page.locator('#render')).toBeDisabled();
  await page.locator('#rgb-red').selectOption('JWST_NIRCAM_F444W');await expect(page.locator('#render')).toBeEnabled();
  await page.locator('#render').click();await expect.poll(()=>submitted).toBeTruthy();
  expect(submitted.instruments).toEqual(['image','rgb','sed']);
  expect(submitted.output).toBe('rgb');
  expect(submitted.rgb_bands).toEqual({red:'JWST_NIRCAM_F444W',green:'JWST_NIRCAM_F200W',blue:'JWST_NIRCAM_F115W'});
  await expect(page.locator('#product-tabs')).toBeVisible();await expect(page.locator('[data-product="rgb"]')).toHaveClass(/active/);
  await page.locator('[data-product="sed"]').click();await expect(page.locator('#spectrum-chart [data-series="smoothed"]')).toBeVisible();
  await page.locator('[data-product="image"]').click();await expect(page.locator('#spectrum-section')).toBeHidden();
});

test('Every configured galaxy can be selected and viewed',async({page})=>{
  await page.goto('/');await expect(page.locator('#render')).toBeEnabled({timeout:120000});
  const config=await(await page.request.get('/api/config')).json();
  const available=config.datasets.filter(x=>x.available),prepared=available.filter(x=>!x.raw),raw=available.filter(x=>x.raw);
  const representative=[...prepared,...[raw[0],raw[Math.floor(raw.length/2)],raw.at(-1)].filter(Boolean)];
  for(const dataset of representative){
    await page.locator('#dataset').selectOption(dataset.id);
    await expect(page.locator('#cloud-loading')).toBeHidden({timeout:120000});
    if(dataset.render_available===false)await expect(page.locator('#render')).toBeDisabled();
    else await expect(page.locator('#render')).toBeEnabled({timeout:120000});
    await expect(page.locator('#scene-name')).toHaveText(dataset.name);
    await expect(page.locator('#error-banner')).toBeHidden();
  }
});

test('MOOSE raw snapshot exposes gas, dust proxy, and guarded on-demand region',async({page})=>{
  await page.goto('/');await expect(page.locator('#cloud-loading')).toBeHidden({timeout:120000});
  const config=await(await page.request.get('/api/config')).json();
  expect(config.datasets.filter(d=>d.raw)).toHaveLength(198);
  await page.locator('#dataset').selectOption('moose-197');
  await expect(page.locator('#cloud-loading')).toBeHidden({timeout:120000});
  await expect(page.locator('#model-region-settings')).toBeVisible();
  await expect(page.locator('#gas-layer-label')).toBeVisible();
  await expect(page.locator('.legend-dot.gas')).toHaveCSS('background-color','rgb(173, 79, 255)');
  await expect(page.locator('#dust-layer-name')).toHaveText('Dust proxy');
  await page.locator('#pick-target').click();await expect(page.locator('#pick-target')).toHaveClass(/active/);
  const viewport=await page.locator('#viewport').boundingBox();
  await page.mouse.click(viewport.x+viewport.width/2,viewport.y+viewport.height/2);
  await expect(page.locator('#pick-target')).not.toHaveClass(/active/);
  await expect(page.locator('#error-banner')).toBeHidden();
  for(const axis of ['x','y','z'])expect(Number.isFinite(Number(await page.locator(`#target-${axis}`).inputValue()))).toBe(true);
  await page.locator('#fov-value').fill('40');await page.locator('#fov-value').press('Tab');
  await page.locator('#model-fov').fill('30');await expect(page.locator('#render')).toBeDisabled();
  await page.locator('#model-fov').fill('50');await expect(page.locator('#render')).toBeEnabled();
  await expect(page.locator('#render-note')).toContainText('On-demand 50 kpc model');
  await page.screenshot({path:'test-results/moose-197-raw.png',fullPage:true});
});

test('Full halo is viewable with megaparsec FOV and a visible render safety limit',async({page})=>{
  const config=await(await page.request.get('/api/config')).json();
  const dataset=config.datasets.find(d=>d.id==='tng100-80-10604-full-halo');
  test.skip(!dataset,'Full-halo dataset is local deployment data');
  await page.goto('/');await expect(page.locator('#cloud-loading')).toBeHidden();
  await page.locator('#dataset').selectOption(dataset.id);
  await expect(page.locator('#cloud-loading')).toBeHidden({timeout:120000});
  await expect(page.locator('#fov-value')).toHaveValue('8000');
  await expect(page.locator('#render')).toBeDisabled();
  await expect(page.locator('#render-note')).toContainText('305 GiB');
  await expect(page.locator('#error-banner')).toBeHidden();
  await page.locator('#camera-mode').click();
  await page.locator('#fov-value').fill('6000');await page.locator('#fov-value').press('Tab');
  await expect(page.locator('#preview-fov')).toHaveText('6000.0 × 6000.0 kpc');
  await page.locator('.target-details summary').click();
  await page.locator('#target-x').fill('-3000');await page.locator('#target-x').press('Tab');
  await expect(page.locator('#target-x')).toHaveValue('-3000');
  await page.locator('#reset-camera').click();await page.locator('#scene-mode').click();
  await page.screenshot({path:'test-results/tng100-full-halo.png',fullPage:true});
  await page.locator('#instrument-sed').check();await expect(page.locator('#render')).toBeDisabled();
  await page.locator('#dataset').selectOption('tng100-80-10604');
  await expect(page.locator('#render')).toBeEnabled();
});

test('Real GPU observation, repeat request caching, camera change and history',async({page})=>{
  test.skip(process.env.SKIRT_VIEWER_REAL_GPU!=='1','Opt in to real catalogue GPU rendering');
  test.setTimeout(900000);
  await page.goto('/');await expect(page.locator('#render')).toBeEnabled({timeout:120000});
  await page.locator('[data-preset="xy"]').click();await page.locator('#pixels').selectOption('256');
  await page.locator('#band').selectOption('JWST_NIRCAM_F150W');
  const responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
  await page.locator('#render').click();const response=await responsePromise;expect(response.status()).toBe(202);
  const first=(await response.json()).job;
  await expect(page.locator('#result-state')).toHaveText('COMPLETED',{timeout:600000});
  await expect(page.locator('#rendered-image')).toBeVisible();await expect(page.locator('#download-fits')).toBeVisible();
  const job=await(await page.request.get(`/api/jobs/${first.id}`)).json();expect(job.gpu_confirmed).toBe(true);expect(job.image.positive_pixels).toBeGreaterThan(100);
  const again=await page.request.post('/api/jobs',{data:first.request});expect((await again.json()).reused).toBe(true);
  await page.locator('#fov-value').fill('36');await page.locator('#fov-value').press('Tab');
  await page.locator('[data-preset="xz"]').click();await page.locator('#band').selectOption('JWST_NIRCAM_F277W');
  await page.locator('.target-details summary').click();await page.locator('#target-x').fill('3');await page.locator('#target-x').press('Tab');
  await expect(page.locator('#stale-notice')).toBeVisible();
  const nextResponse=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
  await page.locator('#render').click();const second=(await(await nextResponse).json()).job;
  expect(second.id).not.toBe(first.id);
  await expect(page.locator('#result-state')).toHaveText('COMPLETED',{timeout:600000});
  await expect(page.locator('#stale-notice')).toBeHidden();
  await page.locator('#camera-mode').click();await page.screenshot({path:'test-results/observatory-real-render.png',fullPage:true});
  await page.reload();await expect(page.locator(`#history [data-job-id="${second.id}"]`)).toBeVisible({timeout:120000});
  await page.locator(`#history [data-job-id="${first.id}"]`).click();await page.locator('#restore-camera').click();
  await expect(page.locator('#band')).toHaveValue('JWST_NIRCAM_F150W');await expect(page.locator('#inclination-value')).toHaveText('0°');
});
