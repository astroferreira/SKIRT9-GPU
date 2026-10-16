#!/usr/bin/env python3
"""Provisional packet-count estimates from independently seeded, identical SKIRT models.

The metric is signal-power-normalized RMS across fixed spatial/spectral regions,
not a guarantee for every faint pixel. Counts are extrapolated, not certified.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from astropy.io import fits

from render_jwst_observation import smooth_spectrum


def metadata(prefix):
    parameters = prefix.with_name(prefix.name + "_parameters.xml")
    log = prefix.with_name(prefix.name + "_log.txt").read_text()
    if f"Finished simulation {prefix.name} " not in log:
        raise ValueError(f"Incomplete run: {prefix}")
    root = ET.parse(parameters).getroot()
    simulation = root.find(".//MonteCarloSimulation")
    packets = float(simulation.attrib.pop("numPackets"))
    seed = int(simulation.find(".//Random").attrib.pop("seed"))
    canonical = ET.canonicalize(ET.tostring(simulation, encoding="unicode"), strip_text=True)
    return {"prefix": str(prefix), "seed": seed, "packets_per_phase": packets,
            "model_signature": hashlib.sha256(canonical.encode()).hexdigest()}, root


def estimate(samples, packets, target):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape[0] < 2 or not np.isfinite(samples).all() or np.any(samples < 0):
        raise ValueError("At least two finite nonnegative realizations are required")
    mean = samples.mean(axis=0)
    variance = samples.var(axis=0, ddof=1)
    noise_power = float(variance.sum())
    # Subtract the noise contribution in the reference mean to avoid artificially
    # high S/N when signal is dominated by noise in both pilot realizations.
    signal_power = float(np.square(mean).sum()) - noise_power / len(samples)
    relative = math.sqrt(noise_power / signal_power) if signal_power > 0 else None
    usable = relative is not None and 0 < relative < 1
    return {"relative_rms": relative,
            "candidate_packets_per_phase": max(1, math.ceil(packets * (relative / target)**2))
            if usable else None,
            "zero_signal_fraction": float(np.mean(mean == 0)),
            "samples_in_region": int(mean.size),
            "signal_power_resolved": signal_power > 0,
            "noise_detected": noise_power > 0,
            "requires_stronger_pilot": not usable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rms", type=float, default=0.01)
    args = parser.parse_args()
    if len(args.prefixes) < 2 or args.target_rms <= 0:
        parser.error("Provide at least two realizations and a positive target RMS")
    loaded = [metadata(prefix.resolve()) for prefix in args.prefixes]
    runs = [item[0] for item in loaded]
    if len({run["seed"] for run in runs}) != len(runs):
        parser.error("Repeated seeds are not independent realizations")
    if len({run["model_signature"] for run in runs}) != 1:
        parser.error("Configurations differ beyond packet count and seed")
    if len({run["packets_per_phase"] for run in runs}) != 1:
        parser.error("This estimator requires equal-count replicates; analyze each count separately")
    packets = runs[0]["packets_per_phase"]
    report = {"status": "Provisional extrapolation; not an optimal-count or convergence certificate",
              "runs": runs, "target_relative_rms": args.target_rms,
              "metric": "sqrt(sum(pixel variance) / noise-corrected signal power)",
              "formula": "N_target = N_pilot * (relative_rms / target_relative_rms)^2",
              "limitations": ["At least four seeds and a second packet count are needed to assess reliability and scaling",
                              "Signal-weighted region RMS does not guarantee S/N in each faint pixel",
                              "Zero-hit pixels and zero measured variance do not demonstrate convergence",
                              "No telescope PSF or detector noise; only explicit spatial binning and ideal R=1000 smoothing",
                              "Primary and secondary packet counts are extrapolated together; nonlinear dust response can alter scaling"],
              "images": [], "spectra": []}
    for frame in loaded[0][1].findall(".//FrameInstrument"):
        view = frame.get("instrumentName")
        bands = [band.get("bandName") for band in frame.findall(".//BroadBand")]
        fov_value, fov_unit = frame.get("fieldOfViewX").split()
        if fov_unit != "pc" or frame.get("fieldOfViewY") != frame.get("fieldOfViewX"):
            raise ValueError("Expected square frames with a field specified in pc")
        fov = float(fov_value) / 1000
        cubes = []
        for prefix in args.prefixes:
            with fits.open(prefix.with_name(f"{prefix.name}_{view}_total.fits"), memmap=True) as hdus:
                cube = np.array(hdus[0].data, dtype=np.float64)
                if cube.shape[0] != len(bands) or not np.isfinite(cube).all() or np.any(cube < 0):
                    raise ValueError("Invalid broadband cube")
                cubes.append(cube)
        stack = np.stack(cubes)
        del cubes
        ny, nx = stack.shape[-2:]
        for binning in (1, 4, 8):
            if nx % binning or ny % binning:
                continue
            data = stack.reshape(len(runs), len(bands), ny // binning, binning,
                                 nx // binning, binning).mean(axis=(3, 5))
            y, x = np.indices(data.shape[-2:], dtype=float)
            radius = np.hypot((x + 0.5) / data.shape[-1] - 0.5,
                              (y + 0.5) / data.shape[-2] - 0.5) * fov
            masks = {"full_field": np.ones_like(radius, dtype=bool), "r0_10kpc": radius < 10,
                     "r10_30kpc": (radius >= 10) & (radius < 30),
                     "r30_75kpc": (radius >= 30) & (radius < 75)}
            for band_index, band in enumerate(bands):
                for name, mask in masks.items():
                    if not mask.any():
                        continue
                    result = estimate(data[:, band_index, mask], packets, args.target_rms)
                    report["images"].append(dict(view=view, band=band, region=name, binning=binning,
                                                 pixel_size_kpc=fov / nx * binning, **result))
        spectra = []
        reference_wave = None
        for prefix in args.prefixes:
            values = np.loadtxt(prefix.with_name(f"{prefix.name}_{view}_nirspec_sed.dat"))
            wave = values[:, 0]
            if reference_wave is not None and not np.array_equal(wave, reference_wave):
                raise ValueError("Spectrum grids differ")
            reference_wave = wave
            spectra.append(smooth_spectrum(wave, values[:, 1], 1000) * wave * 1e-6 / 299792458. / 1e-32)
        for lower, upper in ((0.7, 1.3), (1.3, 2.5), (2.5, 5.3)):
            mask = (wave >= lower) & (wave <= upper)
            report["spectra"].append(dict(view=view, range_micron=[lower, upper], resolution=1000,
                                           **estimate(np.array(spectra)[:, mask], packets, args.target_rms)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
