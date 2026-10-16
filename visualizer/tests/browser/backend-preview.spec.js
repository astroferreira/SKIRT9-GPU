import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';

async function pixels(page){
  const screenshot=await page.locator('#viewport canvas').screenshot();
  return page.evaluate(async encoded=>{
    const image=new Image();image.src=`data:image/png;base64,${encoded}`;await image.decode();
    const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
    const context=canvas.getContext('2d');context.drawImage(image,0,0);
    return [...context.getImageData(0,0,canvas.width,canvas.height).data];
  },screenshot.toString('base64'));
}

test('Backend NVIDIA galaxy frames match browser display without volume downloads',async({page})=>{
  test.skip(process.env.SKIRT_BACKEND_REAL_DATA!=='1','Opt in to NVIDIA rendering of an existing local cached galaxy');
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  const meta=JSON.parse(await readFile('var/catalogues/browser-volumes/7608f26f9eb0c06a2c9ae31fb234913cffe9f75a8e0467b00ecce7cc256df459/volume.json','utf8'));
  await page.route('**/api/datasets/*/preview-volume?**',route=>route.fulfill({json:meta}));
  await page.setViewportSize({width:1100,height:850});
  await page.goto('/');await expect(page.locator('#cloud-loading')).toBeHidden();
  const before=(await(await page.request.get('/api/jobs')).json()).jobs.map(job=>job.id);
  await page.locator('#camera-mode').click();await page.locator('[data-preset="xy"]').click();
  await page.locator('#preview-mode').selectOption('volume');
  await expect(page.locator('#preview-status')).toHaveAttribute('data-state','ready');
  await expect(page.locator('#viewport canvas')).toHaveAttribute('data-volume-samples','8');
  const browser=await pixels(page);
  let downloads=0,frames=0;
  page.on('request',request=>{if(request.url().includes('/api/preview-volumes/'))downloads++;});
  page.on('response',response=>{if(response.url().endsWith('/api/backend-preview/frame')&&response.ok())frames++;});
  await page.locator('#preview-mode').selectOption('backend');
  await expect(page.locator('#preview-status')).toHaveAttribute('data-state','ready');
  await expect(page.locator('#backend-frame-status')).toContainText('NVIDIA GeForce RTX 5090');
  await expect(page.locator('#backend-frame-status')).toContainText('8/8 samples');
  await expect(page.locator('#viewport canvas')).toHaveAttribute('data-volume-samples','8');
  const backend=await pixels(page);
  expect(backend.length).toBe(browser.length);
  let total=0,signal=0;for(let i=0;i<backend.length;i++){if(i%4===3)continue;total+=Math.abs(browser[i]-backend[i]);signal+=backend[i];}
  const mean=total/(backend.length*.75);
  expect(mean).toBeLessThan(2);expect(signal).toBeGreaterThan(10000);
  expect(downloads).toBe(0);expect(frames).toBeGreaterThanOrEqual(8);
  await page.screenshot({path:'test-results/backend-galaxy.png',fullPage:true});
  const lowExposure=page.waitForResponse(response=>response.url().endsWith('/api/backend-preview/frame')&&
    response.request().postDataJSON()?.controls.exposure<=2.1e-7&&response.ok());
  await page.locator('#brightness').fill('-2');await lowExposure;
  await expect(page.locator('#backend-frame-status')).toContainText('8/8 samples');
  const dark=await pixels(page);let darkSignal=0;for(let i=0;i<dark.length;i++)if(i%4!==3)darkSignal+=dark[i];
  expect(darkSignal).toBeLessThan(signal*.2);
  await page.locator('#brightness').fill('0.38');
  const previous=frames;
  await page.locator('#preview-dust').uncheck();
  await expect.poll(()=>frames).toBeGreaterThan(previous);
  await expect(page.locator('#backend-frame-status')).toContainText('8/8 samples');
  await page.locator('[data-preset="xz"]').click();
  await expect.poll(()=>frames).toBeGreaterThan(previous+1);
  await expect(page.locator('#backend-frame-status')).toContainText('8/8 samples');
  await page.locator('#preview-mode').selectOption('diagnostic');
  await expect(page.locator('.stellar-age-key')).toBeVisible();
  const after=(await(await page.request.get('/api/jobs')).json()).jobs.map(job=>job.id);
  expect(after).toEqual(before);expect(errors).toEqual([]);
  console.log(`Backend/browser mean absolute display difference: ${mean.toFixed(4)} / 255; backend frames: ${frames}; volume downloads: ${downloads}`);
});
