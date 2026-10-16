#!/usr/bin/env python3
"""Compare completed, matched SKIRT broadband runs.

This is a regression report for numerical/layout changes.  It compares output
images and timings, but is not a Monte-Carlo noise-parity certificate.
"""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from astropy.io import fits


_STAGES = ("setup", "primary", "secondary", "simulation")
_TIMINGS = {
    "setup": re.compile(r"Finished setup in ([0-9.]+) s\."),
    "primary": re.compile(r"Finished primary emission in ([0-9.]+) s"),
    "secondary": re.compile(r"Finished secondary emission in ([0-9.]+) s"),
    "simulation": re.compile(r"Finished simulation .*? in ([0-9.]+) s"),
}
_MISSING = object()


def canonical_model(path):
    """Return a digest of an expanded parameters file, ignoring writer metadata."""
    root = ET.parse(path).getroot()
    # XmlHierarchyWriter puts these two run-specific fields on the root element.
    root.attrib.pop("producer", None)
    root.attrib.pop("time", None)
    canonical = ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _expected_frames(parameters, prefix):
    root = ET.parse(parameters).getroot()
    default_bands = root.find(".//defaultWavelengthGrid/*/bands")
    expected = {}
    for frame in root.findall(".//FrameInstrument"):
        name = frame.get("instrumentName")
        bands = frame.find("./wavelengthGrid/*/bands")
        if bands is None:
            bands = default_bands
        band_items = list(bands) if bands is not None else []
        try:
            shape = (len(band_items), int(frame.attrib["numPixelsY"]),
                     int(frame.attrib["numPixelsX"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid FrameInstrument declaration in {parameters}") from error
        if not name or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"Invalid FrameInstrument declaration in {parameters}")
        filename = f"{prefix}_{name}_total.fits"
        if filename in expected:
            raise ValueError(f"Duplicate FrameInstrument output name: {filename}")
        expected[filename] = {"shape": shape,
                              "bands": [band.get("bandName") or band.get("filename")
                                        or band.get("type") or band.tag for band in band_items]}
    if not expected:
        raise ValueError(f"No FrameInstrument declarations in {parameters}")
    return expected


def _run(directory):
    directory = Path(directory).resolve()
    parameters = sorted(directory.glob("*_parameters.xml"))
    if len(parameters) != 1:
        raise ValueError(f"Expected exactly one *_parameters.xml in {directory}")
    suffix = "_parameters.xml"
    prefix = parameters[0].name[:-len(suffix)]
    log = directory / f"{prefix}_log.txt"
    if not log.is_file():
        raise ValueError(f"Missing log file: {log}")
    log_text = log.read_text()
    if f"Finished simulation {prefix} " not in log_text:
        raise ValueError(f"Incomplete run: {directory}")
    expected = _expected_frames(parameters[0], prefix)
    images = {path.name: path for path in directory.glob(f"{prefix}_*_total.fits")}
    if set(images) != set(expected):
        raise ValueError(f"Broadband output files do not match FrameInstrument declarations in {directory}")
    timings = {}
    for name, pattern in _TIMINGS.items():
        matches = pattern.findall(log_text)
        timings[name] = float(matches[-1]) if matches else None
    return {"directory": str(directory), "prefix": prefix, "parameters": parameters[0],
            "images": images, "expected_frames": expected,
            "model_signature": canonical_model(parameters[0]), "timings": timings}


def metrics(reference, candidate):
    """Return signal-normalized metrics after rejecting invalid broadband data."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("Mismatched or empty arrays")
    if (not np.isfinite(reference).all() or not np.isfinite(candidate).all()
            or np.any(reference < 0) or np.any(candidate < 0)):
        raise ValueError("Nonfinite or negative image")
    reference_sum = float(reference.sum())
    candidate_sum = float(candidate.sum())
    reference_l2 = float(np.linalg.norm(reference.ravel()))
    peak = float(reference.max())
    if reference_sum <= 0 or candidate_sum <= 0 or reference_l2 <= 0 or peak <= 0:
        raise ValueError("Each band must have a positive sum")
    difference = candidate - reference
    return {"reference_sum": reference_sum, "candidate_sum": candidate_sum,
            "relative_sum_error": abs(candidate_sum / reference_sum - 1.0),
            "relative_l2_error": float(np.linalg.norm(difference.ravel()) / reference_l2),
            "max_peak_normalized_error": float(np.abs(difference).max() / peak)}


def _header_value(header, key):
    return header[key] if key in header else _MISSING


def _check_header(reference, candidate, path):
    keys = ["BUNIT"]
    for axis in range(1, reference[0].data.ndim + 1):
        keys.extend(f"{keyword}{axis}" for keyword in ("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT"))
    for key in keys:
        if _header_value(reference[0].header, key) != _header_value(candidate[0].header, key):
            raise ValueError(f"FITS coordinate/unit mismatch for {path.name}: {key}")


def _wavelengths(hdus, path):
    if len(hdus) < 2 or hdus[1].data is None:
        raise ValueError(f"Missing wavelength-axis extension: {path}")
    values = np.asarray(hdus[1].data.field(0)).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"Invalid wavelength axis: {path}")
    return values


def _compare_image(reference_path, candidate_path, expected):
    rows = []
    with fits.open(reference_path, memmap=True) as reference, fits.open(candidate_path, memmap=True) as candidate:
        ref = np.asarray(reference[0].data, dtype=np.float64)
        cand = np.asarray(candidate[0].data, dtype=np.float64)
        if ref.shape != expected["shape"] or cand.shape != expected["shape"]:
            raise ValueError(f"FITS shape does not match FrameInstrument declaration: {reference_path.name}")
        _check_header(reference, candidate, reference_path)
        ref_wave, cand_wave = _wavelengths(reference, reference_path), _wavelengths(candidate, candidate_path)
        if (_header_value(reference[1].header, "TUNIT1") != _header_value(candidate[1].header, "TUNIT1")
                or not np.array_equal(ref_wave, cand_wave) or ref_wave.size != ref.shape[0]):
            raise ValueError(f"Wavelength-axis mismatch: {reference_path.name}")
        for index, wavelength in enumerate(ref_wave):
            rows.append({"image": reference_path.name, "band": expected["bands"][index], "band_index": index,
                         "wavelength": float(wavelength), **metrics(ref[index], cand[index])})
    return rows


def compare(reference_directory, candidate_directory):
    reference, candidate = _run(reference_directory), _run(candidate_directory)
    if reference["prefix"] != candidate["prefix"]:
        raise ValueError("Runs have different model prefixes")
    if reference["model_signature"] != candidate["model_signature"]:
        raise ValueError("Expanded parameter models differ")
    ref_names = set(reference["images"])
    if ref_names != set(candidate["images"]):
        raise ValueError("Broadband output file sets differ")
    comparisons = []
    for name in sorted(ref_names):
        comparisons.extend(_compare_image(reference["images"][name], candidate["images"][name],
                                          reference["expected_frames"][name]))
    speedups = {stage: (reference["timings"][stage] / candidate["timings"][stage]
                        if reference["timings"][stage] and candidate["timings"][stage] else None)
                for stage in _STAGES}
    return {"purpose": __doc__, "noise_parity_certificate": False,
            "reference": {key: reference[key] for key in ("directory", "prefix", "model_signature", "timings")},
            "candidate": {key: candidate[key] for key in ("directory", "prefix", "model_signature", "timings")},
            "stage_speedups_reference_over_candidate": speedups, "comparisons": comparisons,
            "max_errors": {key: max(row[key] for row in comparisons)
                           for key in ("relative_sum_error", "relative_l2_error", "max_peak_normalized_error")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-identical", nargs="?", type=float, const=0.0, metavar="TOLERANCE",
                        help="Fail if any reported image error exceeds this tolerance (default: exact)")
    args = parser.parse_args()
    if args.require_identical is not None and (not np.isfinite(args.require_identical)
                                                or args.require_identical < 0):
        parser.error("--require-identical tolerance must be finite and nonnegative")
    report = compare(args.reference, args.candidate)
    report["identical_tolerance"] = args.require_identical
    report["passes"] = (None if args.require_identical is None else
                        all(value <= args.require_identical for value in report["max_errors"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("max_errors", "passes")}, indent=2))
    if report["passes"] is False:
        raise RuntimeError("Output regression exceeds --require-identical tolerance")


if __name__ == "__main__":
    main()
