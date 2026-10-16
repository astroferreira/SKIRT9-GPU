import { test, expect } from '@playwright/test';

test('SED mode exposes NIRSpec/MIRI presets and whole-model scope',async({page})=>{
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');await expect(page.locator('#render')).toBeEnabled({timeout:120000});
  await page.locator('#instrument-image').uncheck();await page.locator('#instrument-sed').check();
  await expect(page.locator('#band-settings')).toBeHidden();await expect(page.locator('#pixel-settings')).toBeHidden();
  await expect(page.locator('#fov')).toBeDisabled();await expect(page.locator('#target-x')).toBeDisabled();
  await expect(page.locator('#spectral-mode option')).toHaveCount(11);
  await expect(page.locator('#spectral-resolution')).toHaveValue('100');
  await page.locator('#spectral-mode').selectOption('miri_mrs');await expect(page.locator('#spectral-range')).toContainText('4.9–27.9');
  await expect(page.locator('#spectral-resolution')).toHaveValue('3000');
  await page.locator('#spectral-resolution').fill('750');
  await expect(page.locator('#spectral-range')).toContainText('R ≈ 750');
  await expect(page.locator('#resolution-hint')).toContainText('samples');
  await expect(page.locator('#render')).toBeEnabled();
  await page.locator('#spectral-resolution').fill('0');await expect(page.locator('#render')).toBeDisabled();
  await page.locator('#spectral-resolution').fill('10001');await expect(page.locator('#render')).toBeDisabled();
  await page.locator('#reset-resolution').click();await expect(page.locator('#spectral-resolution')).toHaveValue('3000');
  await page.locator('#camera-mode').click();await expect(page.locator('#frame-guide')).toBeHidden();
  await page.locator('#instrument-image').check();await page.locator('#instrument-sed').uncheck();await expect(page.locator('#fov')).toBeEnabled();await expect(page.locator('#frame-guide')).toBeVisible();
  expect(errors).toEqual([]);
});

test('Custom SED R and source velocities are submitted and restored without a GPU run',async({page})=>{
  // Intercept requests in this browser only; never submit a live observation.
  const job={id:'browser-custom-r',status:'completed',dataset_name:'Resolution test',elapsed_seconds:1,
    phase:'Observation ready',warnings:[],request:null,spectrum_url:'/api/test-custom-spectrum',spectrum:{samples:3}};
  let submitted;
  await page.route('**/api/test-custom-spectrum',route=>route.fulfill({json:{
    wavelength_micron:[4.9,10,27.9],raw_flux:[1,2,1],smoothed_flux:[1,2,1],samples:3,
    mode:{label:'Test MIRI',min_micron:4.9,max_micron:27.9,resolving_power:750},spectral_frame:'observer'}}));
  await page.route('**/api/jobs',async route=>{
    if(route.request().method()==='POST'){
      submitted=route.request().postDataJSON();job.request=submitted;
      await route.fulfill({status:202,json:{job,reused:false}});
    }else await route.fulfill({json:{jobs:job.request?[job]:[]}});
  });
  await page.goto('/');await expect(page.locator('#render')).toBeEnabled();
  const config=await(await page.request.get('/api/config')).json();
  await page.locator('#dataset').selectOption(config.datasets.find(d=>d.available&&d.source_velocities?.available&&d.render_available!==false).id);
  await page.locator('#instrument-image').uncheck();await page.locator('#instrument-sed').check();
  await page.locator('#spectral-mode').selectOption('miri_mrs');
  await page.locator('#spectral-resolution').fill('750');
  await expect(page.locator('#source-velocities')).not.toBeChecked();
  await page.locator('#source-velocities').check();
  await page.locator('#render').click();
  await expect(page.locator('[data-job-id="browser-custom-r"]')).toBeVisible();
  expect(submitted.spectral_resolution).toBe(750);
  expect(submitted.source_velocities).toBe(true);
  await page.locator('#source-velocities').uncheck();
  await page.locator('#spectral-resolution').fill('900');
  await expect(page.locator('#stale-notice')).toBeVisible();
  await page.locator('#restore-camera').click();
  await expect(page.locator('#spectral-resolution')).toHaveValue('750');
  await expect(page.locator('#source-velocities')).toBeChecked();
  await expect(page.locator('#stale-notice')).toBeHidden();
  await expect(page.locator('#error-banner')).toBeHidden();
  await page.screenshot({path:'test-results/source-velocities-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.locator('#source-velocities').scrollIntoViewIfNeeded();
  await page.screenshot({path:'test-results/source-velocities-mobile.png',fullPage:true});
});

test('Spectral references show meanings at saved redshift and remain readable on mobile',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const wave=Array.from({length:500},(_,i)=>5*(28/5)**(i/499));
  const flux=wave.map(w=>1e-15*(1+Math.sin(w)**2));
  const request={dataset_id:'moose-197',output:'sed',instruments:['sed'],spectral_mode:'miri_mrs',spectral_resolution:1000,
    quality:'standard',pixels:512,band:'JWST_NIRCAM_F150W',camera:{inclination:55,azimuth:35,roll:0,fov_kpc:50,target_kpc:[0,0,0]}};
  const pixel='data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
  const job={id:'reference-test',status:'completed',dataset_name:'Reference test galaxy',elapsed_seconds:1,phase:'Observation ready',
    request,primary_product:'sed',spectrum:{samples:500},spectrum_url:'/api/reference-test-spectrum',preview_url:pixel};
  await page.route('**/api/jobs',route=>route.fulfill({json:{jobs:[job]}}));
  await page.route('**/api/reference-test-spectrum',route=>route.fulfill({json:{wavelength_micron:wave,raw_flux:flux,smoothed_flux:flux,
    samples:500,mode:{label:'MRS',min_micron:5,max_micron:28,resolving_power:1000},spectral_frame:'observer',reference_redshift:.5}}));
  await page.goto('/');await expect(page.locator('#spectrum-section')).toBeVisible();
  await expect(page.locator('[data-feature="neii"]')).toHaveAttribute('data-wavelength',String(12.81355*1.5));
  await page.locator('[data-feature-label="neii"]').click();
  await expect(page.locator('#spectrum-feature-details')).toContainText('Singly ionized neon');
  await expect(page.locator('#spectrum-feature-status')).toContainText('not detections');
  await page.locator('[data-feature-group="atomic"]').uncheck();await expect(page.locator('[data-feature="neii"]')).toHaveCount(0);
  await expect(page.locator('[data-feature="pah7.7"]')).toHaveCount(1);
  await page.locator('[data-feature-group="atomic"]').check();
  const hit=await page.locator('[data-plot-hit]').boundingBox();
  const x=w=>hit.x+Math.log(w/5)/Math.log(28/5)*hit.width;
  await page.mouse.move(x(14),hit.y+hit.height/2);await page.mouse.down();await page.mouse.move(x(22),hit.y+hit.height/2,{steps:8});await page.mouse.up();
  await expect(page.locator('[data-feature="h2s7"]')).toHaveCount(0);
  await expect(page.locator('[data-feature="neii"]')).toHaveCount(1);
  await page.locator('#spectrum-reset').click();
  await page.locator('#spectrum-section').screenshot({path:'test-results/spectral-features-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  await expect.poll(()=>page.locator('#spectrum-chart svg').getAttribute('viewBox')).not.toContain('900');
  await page.locator('#spectrum-feature-select').selectOption('silicate97');
  await expect(page.locator('#spectrum-feature-details')).toContainText('silicon-oxygen stretching');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  expect(await page.locator('#spectrum-chart').evaluate(el=>el.scrollWidth<=el.clientWidth)).toBe(true);
  const labels=await page.locator('[data-feature-label]').evaluateAll(nodes=>nodes.map(node=>{
    const r=node.querySelector('rect').getBoundingClientRect(),t=node.querySelector('text').getBoundingClientRect();
    return t.left>=r.left&&t.right<=r.right;
  }));expect(labels.every(Boolean)).toBe(true);
  await page.locator('#spectrum-section').screenshot({path:'test-results/spectral-features-mobile.png'});
  expect(errors).toEqual([]);
});

test('Real NIRSpec and MIRI SED-only observations are plotted, downloadable and restorable',async({page})=>{
  test.skip(process.env.SKIRT_VIEWER_REAL_SED!=='1','Opt in to real GPU SED runs');
  test.setTimeout(1800000);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');await expect(page.locator('#render')).toBeEnabled({timeout:120000});
  await page.locator('#instrument-image').uncheck();await page.locator('#instrument-sed').check();await page.locator('#quality').selectOption('preview');
  const identifiers=[];
  for(const mode of ['nirspec_prism','miri_mrs']){
    await page.locator('#spectral-mode').selectOption(mode);
    const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
    await page.locator('#render').click();const response=await submitted;expect(response.status()).toBe(202);
    const initial=(await response.json()).job;identifiers.push(initial.id);
    console.log(`SED ${mode}: ${initial.id}`);
    await expect(page.locator('#result-state')).toHaveText('COMPLETED',{timeout:850000});
    const job=await(await page.request.get(`/api/jobs/${initial.id}`)).json();expect(job.gpu_confirmed).toBe(true);expect(job.spectrum.positive_bins).toBeGreaterThan(10);
    const ski=await(await page.request.get(`/api/jobs/${job.id}/files/observation.ski`)).text();
    expect((ski.match(/<SEDInstrument /g)||[]).length).toBe(1);expect(ski).not.toContain('<FrameInstrument');
    expect((await page.request.get(`/api/jobs/${job.id}/files/observation_view_total.fits`)).status()).toBe(404);
    for(const key of ['fits_url','raw_url','csv_url','spectrum_url'])expect((await page.request.get(job[key])).status()).toBe(200);
    await expect(page.locator('#spectrum-chart [data-series="smoothed"]')).toBeVisible({timeout:15000});
    await expect(page.locator('#download-sed')).toBeVisible();
    const cached=await page.request.post('/api/jobs',{data:{...initial.request,pixels:1024,band:'SLOAN_SDSS_U'}});expect((await cached.json()).reused).toBe(true);
    await page.locator('#spectrum-raw').check();await page.locator('#spectrum-scale').click();
    const chart=await page.locator('#spectrum-chart svg').boundingBox();
    await page.mouse.move(chart.x+chart.width*.5,chart.y+chart.height*.5);
    await expect(page.locator('#spectrum-readout')).toContainText('μm · λFλ');
    await page.mouse.move(chart.x+chart.width*.25,chart.y+chart.height*.5);await page.mouse.down();await page.mouse.move(chart.x+chart.width*.75,chart.y+chart.height*.5);await page.mouse.up();
    await page.locator('#spectrum-reset').click();
    await page.screenshot({path:`test-results/observatory-${mode}-sed.png`,fullPage:true});
  }
  await page.reload();await expect(page.locator(`[data-job-id="${identifiers[0]}"]`)).toBeVisible({timeout:120000});
  await page.locator(`[data-job-id="${identifiers[0]}"]`).click();await page.locator('#restore-camera').click();
  await expect(page.locator('#instrument-image')).not.toBeChecked();await expect(page.locator('#instrument-sed')).toBeChecked();await expect(page.locator('#spectral-mode')).toHaveValue('nirspec_prism');await expect(page.locator('#stale-notice')).toBeHidden();
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:'test-results/observatory-sed-mobile.png',fullPage:true});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('Existing completed SED displays live reference annotations without rendering',async({page})=>{
  const jobs=(await(await page.request.get('/api/jobs')).json()).jobs;
  const job=jobs.find(j=>j.status==='completed'&&j.request.source_velocities&&
    (j.request.instruments||[j.request.output]).includes('sed'));
  test.skip(!job,'No completed local velocity SED is available');
  await page.route('**/api/jobs',route=>route.fulfill({json:{jobs:[job]}}));
  await page.goto('/');await expect(page.locator('#spectrum-section')).toBeVisible();
  await expect(page.locator('#spectrum-feature-status')).toContainText('not detections');
  expect(await page.locator('[data-feature]').count()).toBeGreaterThan(0);
  const feature=await page.locator('#spectrum-feature-select option').evaluateAll(options=>options.find(option=>option.value)?.value);
  expect(feature).toBeTruthy();
  await page.locator('#spectrum-feature-select').selectOption(feature);
  await expect(page.locator('#spectrum-feature-details')).not.toBeEmpty();
  await page.locator('#spectrum-section').screenshot({path:'test-results/live-sed-features-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  await page.locator('#spectrum-section').screenshot({path:'test-results/live-sed-features-mobile.png'});
  const config=(await(await page.request.get('/api/config')).json()).datasets;
  const unavailable=config.find(d=>d.available&&d.source_velocities?.reason?.includes('PartType4/Velocities'));
  if(unavailable){
    await page.locator('#dataset').selectOption(unavailable.id);
    await expect(page.locator('#source-velocities')).toBeDisabled();
    await expect(page.locator('#source-velocities').locator('..')).toHaveAttribute('title',/Missing snapshot field/);
  }
});
