import {test,expect} from '@playwright/test';

async function settled(page){
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  await expect(page.locator('#viewport canvas')).toHaveAttribute('data-volume-samples','8',{timeout:120000});
}

async function centerColor(page){
  const png=await page.locator('#viewport canvas').screenshot();
  return page.evaluate(async data=>{
    const image=new Image();image.src=`data:image/png;base64,${data}`;await image.decode();
    const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
    const context=canvas.getContext('2d');context.drawImage(image,0,0);
    const pixels=context.getImageData(Math.floor(image.width/2)-24,Math.floor(image.height/2)-24,48,48).data;
    const sum=[0,0,0];for(let i=0;i<pixels.length;i+=4)for(let c=0;c<3;c++)sum[c]+=pixels[i+c];
    return sum.map(value=>value/(pixels.length/4));
  },png.toString('base64'));
}

test('GPU volume shader: extinction, zero dust, refinement, controls and diagnostic fallback',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  const n=8,stars=new Float32Array(n**3*4),young=new Float32Array(n**3*4);
  for(let i=0;i<stars.length;i+=4){stars.set([3,3,3,1],i);}
  await page.route('**/api/datasets/*/preview-volume?**',route=>route.fulfill({json:{
    key:'fixture',size:n,bands:['SLOAN_SDSS_I','SLOAN_SDSS_R','SLOAN_SDSS_G'],redshift:0,voxel_kpc:60/n,
    counts:{stars:512,dust:512},bounds_kpc:[[-30,-30,-30],[30,30,30]],extinction_per_kpc:[.1,.2,.4],
    request:{x:0,y:0,z:0},warnings:['Synthetic homogeneous slab for browser validation.'],
    files:{'stars-dust.bin':'/api/test-volume/stars','young.bin':'/api/test-volume/young'},
  }}));
  await page.route('**/api/test-volume/stars',route=>route.fulfill({body:Buffer.from(stars.buffer),contentType:'application/octet-stream'}));
  await page.route('**/api/test-volume/young',route=>route.fulfill({body:Buffer.from(young.buffer),contentType:'application/octet-stream'}));
  await page.goto('/');await expect(page.locator('#cloud-loading')).toBeHidden({timeout:120000});
  await expect(page.locator('#preview-kernel')).toHaveValue('0.5');
  await expect(page.locator('#brightness')).toHaveAttribute('min','-2');
  await expect(page.locator('#preview-resolution')).toHaveValue('256');
  await page.locator('.shader-controls summary').click();
  await expect(page.locator('#shader-stellar-kernel')).toHaveValue('1');
  await page.locator('#shader-stellar-kernel').fill('0.1');
  await page.locator('#reset-shader').click();
  await expect(page.locator('#shader-stellar-kernel')).toHaveValue('1');
  await page.locator('.shader-controls summary').click();
  await page.locator('#camera-mode').click();await page.locator('[data-preset="xy"]').click();
  await page.locator('#preview-mode').selectOption('volume');
  await expect(page.locator('#preview-status')).toHaveAttribute('data-state','ready');await settled(page);
  const attenuated=await centerColor(page);
  expect(attenuated[0]).toBeGreaterThan(attenuated[1]+10);expect(attenuated[1]).toBeGreaterThan(attenuated[2]+10);
  await page.locator('#preview-dust').uncheck();await settled(page);
  const unattenuated=await centerColor(page);
  expect(unattenuated[0]).toBeGreaterThan(attenuated[0]+15);
  expect(Math.abs(unattenuated[0]-unattenuated[2])).toBeLessThan(5);
  await page.locator('#preview-dust').check();await page.locator('.volume-quality summary').click();await page.locator('#preview-steps').selectOption('384');await settled(page);
  await expect(page.locator('#preview-grid-hint')).toContainText('512 MiB download');
  await expect(page.locator('#preview-grid-hint')).toContainText('1,024 MiB GPU textures');
  await page.locator('#preview-resolution').selectOption('192');
  await expect(page.locator('#preview-grid-hint')).toContainText('216 MiB download');
  await expect(page.locator('#preview-status')).toHaveAttribute('data-state','pending');
  await page.locator('#preview-span').fill('30');
  await expect(page.locator('#preview-grid-hint')).toContainText('156.3 pc per voxel');
  await page.locator('#preview-resolution').selectOption('256');
  await expect(page.locator('#preview-grid-hint')).toContainText('117.2 pc per voxel');
  const requestPromise=page.waitForRequest(request=>request.url().includes('/preview-volume?'));
  await page.locator('#preview-rebuild').click();
  const query=new URL((await requestPromise).url()).searchParams;
  expect(query.get('size')).toBe('256');expect(query.get('span')).toBe('30');
  await expect(page.locator('#preview-status')).toHaveAttribute('data-state','ready');await settled(page);
  await page.locator('#preview-resolution').selectOption('512');
  await expect(page.locator('#preview-grid-hint')).toContainText('High memory');
  await expect(page.locator('#preview-grid-hint')).toContainText('4,096 MiB download');
  await expect(page.locator('#preview-grid-hint')).toContainText('8,192 MiB GPU textures');
  const ultraRequest=page.waitForRequest(request=>request.url().includes('/preview-volume?'));
  await page.locator('#preview-rebuild').click();
  expect(new URL((await ultraRequest).url()).searchParams.get('size')).toBe('512');
  await expect(page.locator('#preview-status')).toHaveAttribute('data-state','ready');await settled(page);
  await page.locator('#preview-kernel').fill('0.1');
  expect(await page.locator('#preview-kernel').evaluate(input=>input.checkValidity())).toBe(true);
  const fine=await centerColor(page);for(let i=0;i<3;i++)expect(Math.abs(fine[i]-attenuated[i])).toBeLessThan(4);
  await page.locator('#preview-dust-scale').fill('0');await settled(page);
  const zero=await centerColor(page);for(let i=0;i<3;i++)expect(Math.abs(zero[i]-unattenuated[i])).toBeLessThan(4);
  await page.locator('#brightness').fill('-2');await settled(page);
  const dark=await centerColor(page);expect(Math.max(...dark)).toBeLessThan(5);
  await page.locator('#brightness').fill('0.38');await settled(page);
  await page.locator('#preview-use-rgb').click();await expect(page.locator('#preview-status')).toHaveAttribute('data-state','pending');
  await expect(page.locator('#preview-red')).toHaveValue('JWST_NIRCAM_F444W');
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.locator('#preview-mode').selectOption('diagnostic');await expect(page.locator('.stellar-age-key')).toBeVisible();
  expect(errors).toEqual([]);
});

test('Full-catalogue band volume renders real local data without a SKIRT job',async({page})=>{
  test.skip(process.env.SKIRT_PREVIEW_REAL_DATA!=='1','Opt in to local full-catalogue volume preparation');
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  await page.goto('/');await expect(page.locator('#cloud-loading')).toBeHidden({timeout:120000});
  const before=(await(await page.request.get('/api/jobs')).json()).jobs.map(job=>job.id);
  await page.locator('#camera-mode').click();await page.locator('[data-preset="xy"]').click();
  await page.locator('#preview-mode').selectOption('volume');
  await expect(page.locator('#preview-status')).toHaveAttribute('data-state','ready',{timeout:120000});await settled(page);
  const attenuated=await centerColor(page);
  expect(Math.max(...attenuated)).toBeGreaterThan(5);
  await page.screenshot({path:'test-results/volume-real-dust.png',fullPage:true});
  await page.locator('#preview-dust').uncheck();await settled(page);
  const clear=await centerColor(page);expect(clear.reduce((a,b)=>a+b,0)).toBeGreaterThan(attenuated.reduce((a,b)=>a+b,0));
  await page.screenshot({path:'test-results/volume-real-clear.png',fullPage:true});
  const after=(await(await page.request.get('/api/jobs')).json()).jobs.map(job=>job.id);expect(after).toEqual(before);
  expect(errors).toEqual([]);
});
