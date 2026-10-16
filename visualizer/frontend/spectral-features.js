// Reference positions, not a line-fitting or detection catalogue.
const SDSS='https://classic.sdss.org/dr7/algorithms/linestable.php';
const ISO='https://www.ipac.caltech.edu/iso/sws/atomic.html';
const H2='https://raw.githubusercontent.com/PAHFIT/pahfit/master/pahfit/packs/science/classic.yaml';
const PAH='https://arxiv.org/abs/astro-ph/0610913';
const HYDROGEN='https://ned.ipac.caltech.edu/level5/March01/Dinerstein2/Diner2.html';
export const FEATURE_GROUPS={
  atomic:{label:'Ionized gas',color:'#376caa'},
  molecular:{label:'Molecular H2',color:'#9a4667'},
  pah:{label:'PAH bands',color:'#8b650b'},
  dust:{label:'Silicate dust',color:'#725ca5'},
};
const feature=(id,label,rest,group,meaning,source,priority=1)=>({id,label,rest,group,meaning,source,priority});
const recombination='Hydrogen recombination in ionized gas. Often powered by young stars; flux also depends on attenuation and the ionizing source.';
const molecular='Rotational emission from warm molecular hydrogen. UV heating and shocks can both excite it; one line alone does not distinguish them.';
export const SPECTRAL_FEATURES=[
  feature('oii3727','[O II] 3727',.3727092,'atomic','Singly ionized oxygen. Sensitive to ionized gas, abundance and excitation; not a unique star-formation measure.',SDSS),
  feature('oii3729','[O II] 3729',.3729875,'atomic','The second oxygen doublet component. Its ratio with [O II] 3727 is sensitive to electron density.',SDSS,0),
  feature('hgamma','H-gamma',.434168,'atomic',recombination,SDSS),
  feature('hbeta','H-beta',.486268,'atomic',recombination+' Its ratio with H-alpha can constrain attenuation under an assumed recombination model.',SDSS,3),
  feature('oiii4960','[O III] 4960',.4960295,'atomic','Doubly ionized oxygen in ionized gas; the weaker member of the optical [O III] pair.',SDSS,0),
  feature('oiii5008','[O III] 5008',.500824,'atomic','Doubly ionized oxygen. Sensitive to excitation, temperature and abundance; not by itself proof of an active nucleus.',SDSS,3),
  feature('nii6549','[N II] 6549',.654986,'atomic','Singly ionized nitrogen. A weaker line near H-alpha, sensitive to abundance and ionization conditions.',SDSS,0),
  feature('halpha','H-alpha',.656461,'atomic',recombination,SDSS,4),
  feature('nii6585','[N II] 6585',.658527,'atomic','Singly ionized nitrogen. Often blended with H-alpha at low resolution; line ratios depend on abundance and excitation.',SDSS,2),
  feature('sii6718','[S II] 6718',.671829,'atomic','Singly ionized sulfur from low-ionization gas. The sulfur doublet is an electron-density diagnostic.',SDSS),
  feature('pabeta','Pa-beta',1.282,'atomic',recombination+' An infrared Paschen-series line, less attenuated than the Balmer lines.',HYDROGEN,2),
  feature('paalpha','Pa-alpha',1.8756,'atomic',recombination+' An infrared Paschen-series line.',HYDROGEN,3),
  feature('brgamma','Br-gamma',2.166,'atomic',recombination+' An infrared Brackett-series line.',HYDROGEN,2),
  feature('bralpha','Br-alpha',4.052,'atomic',recombination+' The strongest Brackett-series transition.',HYDROGEN,2),
  feature('arii','[Ar II]',6.985274,'atomic','Singly ionized argon: a fine-structure line of relatively low-ionization gas.',ISO),
  feature('ariii','[Ar III]',8.99103,'atomic','Doubly ionized argon. Compared with [Ar II], it constrains the gas excitation.',ISO),
  feature('siv','[S IV]',10.5105,'atomic','Triply ionized sulfur. Requires more energetic ionizing radiation than [S III].',ISO,2),
  feature('neii','[Ne II]',12.81355,'atomic','Singly ionized neon: ionized-gas emission often associated with star-forming regions.',ISO,4),
  feature('nev','[Ne V]',14.32,'atomic','Four-times ionized neon. Associated with very hard radiation or fast shocks; an AGN indicator in context, not a detection claim.',ISO,1),
  feature('neiii','[Ne III]',15.555,'atomic','Doubly ionized neon. The [Ne III]/[Ne II] ratio probes excitation and ionizing-spectrum hardness.',ISO,3),
  feature('siii','[S III]',18.7129,'atomic','Doubly ionized sulfur in ionized gas. Its ratio to the 33.5 micron line is density-sensitive.',ISO,2),
  feature('oiv','[O IV]',25.913,'atomic','Triply ionized oxygen. Hard radiation and shocks can contribute; its presence alone does not establish an AGN.',ISO,2),
  ...[[7,5.5115],[6,6.1088],[5,6.9091],[4,8.0258],[3,9.6649],[2,12.2785],[1,17.0346],[0,28.2207]].map(([j,w])=>
    feature(`h2s${j}`,`H2 S(${j})`,w,'molecular',molecular,H2,j===1||j===3?2:1)),
  ...[[6.2,'6.2'],[7.7,'7.7'],[8.6,'8.6'],[11.3,'11.3'],[12.7,'12.7'],[17,'17']].map(([w,label])=>
    feature(`pah${label}`,`PAH ${label}`,w,'pah','Broad aromatic dust emission complex from polycyclic aromatic hydrocarbons heated by radiation. Its strength depends on grain properties and environment; not a narrow gas line.',PAH,w===7.7||w===11.3?3:2)),
  feature('silicate97','Silicate 9.7',9.7,'dust','Broad silicon-oxygen stretching feature. Can appear in emission or absorption depending on dust temperature and viewing geometry; not a narrow line.',PAH,3),
  feature('silicate18','Silicate 18',18,'dust','Broad oxygen-silicon-oxygen bending feature. Its profile depends on grain properties and radiative-transfer geometry.',PAH,2),
];

export function referenceFeatures(data,groups,range){
  const z=data.spectral_frame==='rest'?0:data.reference_redshift;
  if(typeof z!=='number'||!Number.isFinite(z)||z<0)return [];
  return SPECTRAL_FEATURES.filter(f=>groups.has(f.group)).map(f=>({...f,wavelength:f.rest*(1+z)}))
    .filter(f=>Math.log10(f.wavelength)>=range[0]&&Math.log10(f.wavelength)<=range[1])
    .sort((a,b)=>a.wavelength-b.wavelength);
}

export function placeFeatureLabels(features,sx,left,right,lanes=3){
  const occupied=Array.from({length:lanes},()=>[]),result=[];
  for(const f of [...features].sort((a,b)=>b.priority-a.priority||a.wavelength-b.wavelength)){
    const width=f.label.length*6.6+12;
    if(width>right-left)continue;
    const x=clamp(sx(f.wavelength),left+width/2,right-width/2),lo=x-width/2,hi=x+width/2;
    const lane=occupied.findIndex(row=>row.every(([a,b])=>hi+5<a||lo-5>b));
    if(lane<0)continue;
    occupied[lane].push([lo,hi]);result.push({...f,x,width,lane});
  }
  return result;
}
const clamp=(x,low,high)=>Math.min(high,Math.max(low,x));
