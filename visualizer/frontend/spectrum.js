import { FEATURE_GROUPS, referenceFeatures, placeFeatureLabels } from './spectral-features.js';

const NS='http://www.w3.org/2000/svg';
const clamp=(x,low,high)=>Math.min(high,Math.max(low,x));
const tick=(x)=>x===0?'0':Math.abs(x)<.001||Math.abs(x)>=10000?x.toExponential(1):Number(x.toPrecision(3)).toString();

export class SpectrumPlot {
  constructor(container,readout,reference={}) {
    this.container=container;this.readout=readout;this.logFlux=true;this.showRaw=false;
    this.reference=reference;this.featureGroups=new Set(Object.keys(FEATURE_GROUPS));this.selectedFeature=null;this.features=[];
    this.width=900;this.height=390;this.left=72;this.right=875;this.top=82;this.bottom=336;
    reference.select?.addEventListener('change',()=>this.selectFeature(reference.select.value));
    this.resizeObserver=new ResizeObserver(entries=>{
      const width=Math.round(entries[0].contentRect.width);
      if(width>0&&width!==this.width){this.width=width;this.right=width-18;this.draw();}
    });
    this.resizeObserver.observe(container);
  }
  setData(data) {
    this.data=data;this.selectedFeature=null;this.reset();
  }
  reset() {
    if(!this.data)return;
    const x=this.data.wavelength_micron;
    this.range=[Math.log10(x[0]),Math.log10(x.at(-1))];this.draw();
  }
  setLog(value){this.logFlux=value;this.draw();}
  setRaw(value){this.showRaw=value;this.draw();}
  setFeatureGroup(group,enabled){if(enabled)this.featureGroups.add(group);else this.featureGroups.delete(group);this.draw();}
  selectFeature(id){this.selectedFeature=id||null;this.updateReference();}
  updateReference(){
    if(!this.data)return;
    const {select,details,status}=this.reference;
    const feature=this.features.find(f=>f.id===this.selectedFeature);
    if(!feature)this.selectedFeature=null;
    if(select){
      const empty=document.createElement('option');empty.value='';empty.textContent='Select a reference feature';
      select.replaceChildren(empty);
      for(const f of this.features){const option=document.createElement('option');option.value=f.id;option.textContent=`${f.label} · ${f.wavelength.toFixed(4)} μm`;select.append(option);}
      select.value=this.selectedFeature||'';select.disabled=!this.features.length;
    }
    if(status){
      const known=this.data.spectral_frame==='rest'||Number.isFinite(this.data.reference_redshift);
      status.textContent=!known?'Reference positions unavailable: observer redshift is not recorded.':
        `${this.features.length} reference features in view · ${this.data.spectral_frame==='rest'?'rest frame':`observer z = ${Number(this.data.reference_redshift).toPrecision(4)}`} · not detections`;
    }
    if(details){
      details.replaceChildren();
      if(feature){
        const title=document.createElement('strong');title.textContent=`${feature.label} · rest ${feature.rest} μm · plotted ${feature.wavelength.toFixed(5)} μm`;
        const meaning=document.createElement('p');meaning.textContent=feature.meaning;
        const link=document.createElement('a');link.href=feature.source;link.target='_blank';link.rel='noopener noreferrer';link.textContent='Scientific reference';
        details.append(title,meaning,link);
      }else{
        const note=document.createElement('p');note.textContent='Reference positions do not establish line strength or detection. PAH and silicate markers denote broad bands. Source velocities can shift line peaks away from the systemic reference.';details.append(note);
      }
    }
    this.svg?.querySelectorAll('[data-feature-label]').forEach(node=>node.setAttribute('aria-pressed',String(node.dataset.featureLabel===this.selectedFeature)));
  }
  node(tag,attrs={},parent=this.svg,text=null) {
    const node=document.createElementNS(NS,tag);
    for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));
    if(text!==null)node.textContent=text;
    parent.append(node);return node;
  }
  draw() {
    if(!this.data)return;
    const {wavelength_micron:x,raw_flux:raw,smoothed_flux:smooth}=this.data;
    const indices=x.map((v,i)=>i).filter(i=>Math.log10(x[i])>=this.range[0]&&Math.log10(x[i])<=this.range[1]);
    let low=Infinity,high=-Infinity;
    for(const i of indices)for(const value of this.showRaw?[smooth[i],raw[i]]:[smooth[i]]){
      if(this.logFlux&&value<=0)continue;
      const y=this.logFlux?Math.log10(value):value;low=Math.min(low,y);high=Math.max(high,y);
    }
    if(!Number.isFinite(low)){low=this.logFlux?-20:0;high=this.logFlux?-19:1;}
    if(low===high){const pad=this.logFlux?.5:Math.max(high*.1,1e-30);low-=pad;high+=pad;}
    const pad=(high-low)*.07;
    this.yRange=[this.logFlux?low-pad:Math.max(0,low-pad),high+pad];
    this.sx=value=>this.left+(Math.log10(value)-this.range[0])/(this.range[1]-this.range[0])*(this.right-this.left);
    this.sy=value=>this.bottom-((this.logFlux?Math.log10(value):value)-this.yRange[0])/(this.yRange[1]-this.yRange[0])*(this.bottom-this.top);
    this.features=referenceFeatures(this.data,this.featureGroups,this.range);
    this.svg=document.createElementNS(NS,'svg');
    this.svg.setAttribute('viewBox',`0 0 ${this.width} ${this.height}`);
    this.svg.setAttribute('role','img');this.svg.setAttribute('aria-label',`${this.data.mode.label} integrated SED, wavelength in micron and lambda F lambda in watts per square metre`);
    this.node('title',{},this.svg,`${this.data.mode.label} integrated spectral energy distribution`);
    const ticks=this.width<500?3:5;
    for(let i=0;i<=ticks;i++){
      const px=this.left+(this.right-this.left)*i/ticks;
      const py=this.bottom-(this.bottom-this.top)*i/ticks;
      this.node('line',{x1:px,x2:px,y1:this.top,y2:this.bottom,stroke:'#e3e9dc','stroke-width':.6});
      this.node('line',{x1:this.left,x2:this.right,y1:py,y2:py,stroke:'#e3e9dc','stroke-width':.6});
      this.node('text',{x:px,y:this.bottom+21,'text-anchor':i===ticks?'end':i===0?'start':'middle'},this.svg,tick(10**(this.range[0]+(this.range[1]-this.range[0])*i/ticks)));
      const y=this.yRange[0]+(this.yRange[1]-this.yRange[0])*i/ticks;
      this.node('text',{x:this.left-12,y:py+3,'text-anchor':'end'},this.svg,tick(this.logFlux?10**y:y));
    }
    this.node('text',{x:(this.left+this.right)/2,y:this.height-8,'text-anchor':'middle',class:'axis-label'},this.svg,`${this.data.spectral_frame==='observer'?'Observer':'Rest'} wavelength (μm)`);
    this.node('text',{x:15,y:(this.top+this.bottom)/2,transform:`rotate(-90 15 ${(this.top+this.bottom)/2})`,'text-anchor':'middle',class:'axis-label'},this.svg,'λFλ (W m⁻²)');
    for(const f of this.features){
      const color=FEATURE_GROUPS[f.group].color,px=this.sx(f.wavelength);
      this.node('line',{x1:px,x2:px,y1:this.top,y2:this.bottom,stroke:color,'stroke-width':.8,'stroke-dasharray':f.group==='pah'||f.group==='dust'?'2 5':'5 4',opacity:.5,'data-feature':f.id,'data-wavelength':f.wavelength});
    }
    const featureLabels=placeFeatureLabels(this.features,this.sx,this.left,this.right);
    for(const f of featureLabels){
      this.node('path',{d:`M${f.x},${26+f.lane*23}L${this.sx(f.wavelength)},${this.top-3}`,stroke:FEATURE_GROUPS[f.group].color,'stroke-width':.6,fill:'none',opacity:.5,'pointer-events':'none'});
    }
    for(const f of featureLabels){
      const color=FEATURE_GROUPS[f.group].color,y=18+f.lane*23;
      const label=this.node('g',{'data-feature-label':f.id,role:'button',tabindex:0,'aria-label':`${f.label}, rest ${f.rest} micron. ${f.meaning}`,cursor:'pointer'});
      this.node('title',{},label,`${f.label} · rest ${f.rest} μm · plotted ${f.wavelength.toFixed(5)} μm\n${f.meaning}`);
      this.node('rect',{x:f.x-f.width/2,y:y-13,width:f.width,height:21,fill:'#fafbf7',rx:2},label);
      this.node('text',{x:f.x,y,'text-anchor':'middle',style:`fill:${color}`},label,f.label);
      label.addEventListener('click',()=>this.selectFeature(f.id));
      label.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();this.selectFeature(f.id);}});
    }
    const path=values=>{
      let drawing=false,points=[];
      for(const i of indices){
        if(this.logFlux&&values[i]<=0){drawing=false;continue;}
        points.push(`${drawing?'L':'M'}${this.sx(x[i]).toFixed(2)},${this.sy(values[i]).toFixed(2)}`);drawing=true;
      }
      return points.join(' ');
    };
    if(this.showRaw)this.node('path',{d:path(raw),fill:'none',stroke:'#9bbbad','stroke-width':.7});
    this.node('path',{d:path(smooth),fill:'none',stroke:'#245f51','stroke-width':1.2,'data-series':'smoothed'});
    this.crosshair=this.node('line',{x1:0,x2:0,y1:this.top,y2:this.bottom,stroke:'#779781','stroke-dasharray':'3 4',visibility:'hidden'});
    this.selection=this.node('rect',{y:this.top,height:this.bottom-this.top,x:0,width:0,fill:'#67a07d',opacity:.14});
    const hit=this.node('rect',{x:this.left,y:this.top,width:this.right-this.left,height:this.bottom-this.top,fill:'transparent',cursor:'crosshair','data-plot-hit':true});
    this.container.replaceChildren(this.svg);
    this.updateReference();
    let drag=null;
    const local=e=>clamp((e.clientX-this.svg.getBoundingClientRect().left)/this.svg.getBoundingClientRect().width*this.width,this.left,this.right);
    hit.addEventListener('pointerdown',e=>{drag={start:local(e),id:e.pointerId};hit.setPointerCapture(e.pointerId);});
    hit.addEventListener('pointermove',e=>{
      const px=local(e),logW=this.range[0]+(px-this.left)/(this.right-this.left)*(this.range[1]-this.range[0]);
      const w=10**logW;let l=0,r=x.length-1;
      while(l<r){const m=(l+r)>>1;if(x[m]<w)l=m+1;else r=m;}
      const i=l>0&&Math.abs(x[l-1]-w)<Math.abs(x[l]-w)?l-1:l;
      this.crosshair.setAttribute('x1',this.sx(x[i]));this.crosshair.setAttribute('x2',this.sx(x[i]));this.crosshair.setAttribute('visibility','visible');
      this.readout.textContent=`λ ${x[i].toFixed(4)} μm · λFλ ${smooth[i].toExponential(4)} W/m² · raw ${raw[i].toExponential(4)} W/m²`;
      const nearest=this.features.reduce((best,f)=>!best||Math.abs(this.sx(f.wavelength)-px)<Math.abs(this.sx(best.wavelength)-px)?f:best,null);
      if(nearest&&Math.abs(this.sx(nearest.wavelength)-px)<8)this.readout.textContent+=` · ${nearest.label} reference`;
      if(drag){this.selection.setAttribute('x',Math.min(drag.start,px));this.selection.setAttribute('width',Math.abs(px-drag.start));}
    });
    hit.addEventListener('pointerup',e=>{
      if(!drag)return;const end=local(e),start=drag.start;drag=null;
      if(Math.abs(end-start)>12){
        const from=px=>this.range[0]+(px-this.left)/(this.right-this.left)*(this.range[1]-this.range[0]);
        const next=[from(Math.min(start,end)),from(Math.max(start,end))];
        const count=x.filter(v=>Math.log10(v)>=next[0]&&Math.log10(v)<=next[1]).length;
        if(count>=3){this.range=next;this.draw();return;}
      }
      this.selection.setAttribute('width',0);
      if(Math.abs(end-start)<=12){
        const nearby=this.features.filter(f=>Math.abs(this.sx(f.wavelength)-end)<8).sort((a,b)=>Math.abs(this.sx(a.wavelength)-end)-Math.abs(this.sx(b.wavelength)-end));
        if(nearby.length)this.selectFeature(nearby[0].id);
      }
    });
    hit.addEventListener('pointercancel',()=>{drag=null;this.selection.setAttribute('width',0);});
    hit.addEventListener('pointerleave',()=>{if(!drag)this.crosshair.setAttribute('visibility','hidden');});
  }
}
