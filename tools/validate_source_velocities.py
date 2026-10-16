"""Matched CPU/GPU Doppler regressions with analytic narrow-line sources."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
C = 299792.458
LINE = .121567


def configuration(template, folder, velocity, weights, dispersion, dust_mass, packets, enabled=True):
    tree = ET.parse(template)
    root = tree.getroot()
    sim = root.find('.//MonteCarloSimulation')
    sim.set('simulationMode', 'ExtinctionOnly'); sim.set('numPackets', str(packets))
    sim.set('iterateSecondaryEmission', 'false')
    cosmo = root.find('.//cosmology'); cosmo.clear(); cosmo.set('type', 'Cosmology')
    ET.SubElement(cosmo, 'LocalUniverseCosmology')
    sources = root.find('.//sources'); first = copy.deepcopy(sources[0]); sources.clear(); sources.set('type','Source')
    sources.append(first)
    first.set('filename',str(folder/'sources.dat')); first.set('wavelengthBias','0')
    first.set('importVelocity',str(enabled).lower()); first.set('importVelocityDispersion',str(dispersion > 0).lower())
    family = first.find('sedFamily'); family.clear(); family.set('type','SEDFamily')
    ET.SubElement(family,'LyaGaussianSEDFamily')
    ss = root.find('.//SourceSystem'); ss.set('minWavelength','.117 micron'); ss.set('maxWavelength','.126 micron')
    table = []
    for i, (v, w) in enumerate(zip(velocity, weights)):
        row = [(-1)**i,0,0,.01]
        if enabled: row.extend([0,0,v])
        if dispersion > 0: row.append(dispersion)
        row.extend([w*1e30, 10.])
        table.append(row)
    np.savetxt(folder/'sources.dat', table, fmt='%.17g')
    ms = root.find('.//MediumSystem')
    ms.remove(ms.find('dustEmissionOptions'))
    ms.find('.//RadiationFieldOptions').set('storeRadiationField','false')
    mesh = root.find('.//VoronoiMeshMedium'); mesh.set('filename',str(folder/'dust.dat'))
    for node in (mesh,root.find('.//VoronoiMeshSpatialGrid')):
        for axis in 'XYZ': node.set('min'+axis,'-10 pc'); node.set('max'+axis,'10 pc')
    np.savetxt(folder/'dust.dat', [[x,y,z,dust_mass] for x in (-5,5) for y in (-5,5) for z in (-5,5)])
    system = root.find('.//InstrumentSystem'); system.clear()
    default = ET.SubElement(system,'defaultWavelengthGrid',type='WavelengthGrid')
    ET.SubElement(default,'ResolutionWavelengthGrid',minWavelength='.1209 micron',maxWavelength='.1223 micron',resolution='100000')
    instruments = ET.SubElement(system,'instruments',type='Instrument')
    for name, inclination in [('toward',0),('side',90),('away',180)]:
        ET.SubElement(instruments,'SEDInstrument',instrumentName=name,distance='10 Mpc',inclination=f'{inclination} deg',
                      azimuth='0 deg',recordComponents='false',recordStatistics='false')
    for probes in root.findall('.//probes'): probes.clear(); probes.set('type','Probe')
    ET.indent(tree,space='    '); tree.write(folder/'model.ski',encoding='utf-8',xml_declaration=True)


def moments(path):
    wave, flux = np.loadtxt(path,unpack=True)
    v = C*(wave/LINE-1)
    energy = flux*np.gradient(np.log(wave))
    centroid = np.average(v,weights=energy)
    return dict(centroid_km_s=float(centroid),sigma_km_s=float(np.sqrt(np.average((v-centroid)**2,weights=energy))),
                flux=float(energy.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default=str(ROOT.parent/'release/SKIRT/main/skirtgpu'))
    parser.add_argument('--cpu',default=str(ROOT.parent/'cpu-reference/release/SKIRT/main/skirt'))
    parser.add_argument('--packets',type=int,default=1000000)
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    cases = dict(zero=([0,0],[1,3],0,1e-20), bulk=([300,300],[1,3],0,1e-20),
                 split=([-250,250],[1,3],0,1e-20), dispersion=([0,0],[1,3],40,1e-20),
                 dusty=([-250,250],[1,3],0,1.))
    results = {}
    for case, (velocities, weights, dispersion, dust_mass) in cases.items():
        results[case] = {}
        for backend in ('cpu','gpu'):
            folder=args.output/case/backend; folder.mkdir(parents=True,exist_ok=True)
            configuration(args.template,folder,velocities,weights,dispersion,dust_mass,args.packets)
            env = {k:v for k,v in os.environ.items() if not k.startswith('SKIRTGPU')}
            env.update(SKIRTGPU_SOURCE_LAUNCH='0',SKIRTGPU_PRECOMPUTE_SOURCE_SPECTRA='0',
                       SKIRTGPU_PRIMARY_LIFECYCLE_THREADS='1',SKIRTGPU_BATCH_FIRST_FORCED_CHUNK='262144',
                       SKIRTGPU_RESIDENT_MIN_PACKETS='1',SKIRTGPU_PROFILE_BATCH='1',SKIRTGPU_PROFILE_BATCH_LIMIT='10')
            command=[getattr(args,backend),*(['-g'] if backend=='gpu' else []),'-t','1','-o',str(folder),str(folder/'model.ski')]
            start=time.monotonic()
            with (folder/'console.log').open('w') as log:
                run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            if run.returncode: raise RuntimeError((folder/'console.log').read_text()[-4000:])
            metrics={name:moments(folder/f'model_{name}_sed.dat') for name in ('toward','side','away')}
            resident='resident_lifecycle=yes' in (folder/'model_log.txt').read_text()
            if backend=='gpu' and not resident: raise AssertionError('GPU resident path was not used')
            results[case][backend]=dict(seconds=time.monotonic()-start,metrics=metrics,resident=resident)
            print(case,backend,results[case][backend],flush=True)
        for name in ('toward','side','away'):
            cpu=results[case]['cpu']['metrics'][name]; gpu=results[case]['gpu']['metrics'][name]
            assert abs(cpu['centroid_km_s']-gpu['centroid_km_s'])<2., (case,name,cpu,gpu)
            assert abs(cpu['sigma_km_s']-gpu['sigma_km_s'])<max(1.,.02*cpu['sigma_km_s']), (case,name,cpu,gpu)
            assert abs(cpu['flux']/gpu['flux']-1)<.015, (case,name,cpu,gpu)
            if case!='dusty':
                los=np.cos(np.radians({'toward':0,'side':90,'away':180}[name]))
                centres=-np.asarray(velocities)*los
                expected=float(np.average(centres,weights=weights))
                sigma=np.sqrt(100.+dispersion**2+np.average((centres-expected)**2,weights=weights))
                assert abs(gpu['centroid_km_s']-expected)<2., (case,name,gpu,expected)
                assert abs(gpu['sigma_km_s']-sigma)<max(1.,.02*sigma), (case,name,gpu,sigma)
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('All analytic and CPU/GPU source-velocity checks passed',flush=True)


if __name__=='__main__': main()
