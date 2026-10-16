#!/usr/bin/env python3
"""Check same-model, same-seed JWST outputs after a numerical/layout change.

This is a regression comparison, not an independent-noise or convergence test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from estimate_jwst_packet_noise import metadata


def metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("Mismatched or empty arrays")
    if (not np.isfinite(reference).all() or not np.isfinite(candidate).all()
            or np.any(reference < 0) or np.any(candidate < 0)):
        raise ValueError("Nonfinite or negative output")
    peak = float(reference.max())
    if peak <= 0:
        raise ValueError("Empty reference signal")
    residual = (candidate - reference) / peak
    scaled_reference = reference / peak
    return {
        "max_peak_normalized_error": float(np.abs(residual).max()),
        "relative_l2_error": float(np.linalg.norm(residual.ravel())
                                   / np.linalg.norm(scaled_reference.ravel())),
        "relative_sum_error": float(abs(candidate.sum() / reference.sum() - 1)),
        "identical_fraction": float(np.mean(reference == candidate)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Positive finite tolerance required")
    prefixes = [args.reference.resolve(), args.candidate.resolve()]
    loaded = [metadata(prefix) for prefix in prefixes]
    runs = [value[0] for value in loaded]
    for key in ("model_signature", "seed", "packets_per_phase"):
        if runs[0][key] != runs[1][key]:
            raise ValueError(f"Runs differ in {key}; this comparison requires a matched seed and configuration")
    report = {"purpose": __doc__, "runs": runs, "tolerance": args.tolerance, "comparisons": []}
    frames = loaded[0][1].findall(".//FrameInstrument")
    if not frames:
        raise ValueError("No frame instruments")
    for frame in frames:
        view = frame.get("instrumentName")
        paths = [prefix.with_name(f"{prefix.name}_{view}_total.fits") for prefix in prefixes]
        bands = [band.get("bandName") for band in frame.findall(".//BroadBand")]
        expected_shape = (len(bands), int(frame.get("numPixelsY")), int(frame.get("numPixelsX")))
        with fits.open(paths[0], memmap=True) as reference, fits.open(paths[1], memmap=True) as candidate:
            if reference[0].data.shape != expected_shape or candidate[0].data.shape != expected_shape:
                raise ValueError(f"Unexpected cube shape for {view}")
            for key in ("BUNIT", "CDELT1", "CDELT2", "CRPIX1", "CRPIX2"):
                if reference[0].header.get(key) != candidate[0].header.get(key):
                    raise ValueError(f"FITS header mismatch: {view} {key}")
            if not np.array_equal(reference[1].data.field(0), candidate[1].data.field(0)):
                raise ValueError(f"Wavelength-axis mismatch: {view}")
            for i, band in enumerate(bands):
                report["comparisons"].append({"view": view, "product": "image", "band": band,
                    **metrics(reference[0].data[i], candidate[0].data[i])})
        paths = [prefix.with_name(f"{prefix.name}_{view}_nirspec_sed.dat") for prefix in prefixes]
        reference, candidate = [np.loadtxt(path) for path in paths]
        if reference.shape != candidate.shape or not np.array_equal(reference[:, 0], candidate[:, 0]):
            raise ValueError(f"Spectrum wavelength-axis mismatch: {view}")
        report["comparisons"].append({"view": view, "product": "spectrum",
                                     **metrics(reference[:, 1:], candidate[:, 1:])})
    keys = ("max_peak_normalized_error", "relative_l2_error", "relative_sum_error")
    report["max_errors"] = {key: max(row[key] for row in report["comparisons"]) for key in keys}
    report["passes"] = all(value <= args.tolerance for value in report["max_errors"].values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("max_errors", "passes")}, indent=2))
    if not report["passes"]:
        raise RuntimeError("Output regression exceeds the configured tolerance")


if __name__ == "__main__":
    main()
