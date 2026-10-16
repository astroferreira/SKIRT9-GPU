#!/usr/bin/env python3
"""Verify GPU seed sensitivity on a real SKIRT model; this is not a speed benchmark."""

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from astropy.io import fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ski", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    images = []
    for label, seed in (("a", 12345), ("b", 67890), ("repeat_a", 12345)):
        directory = args.output / label
        directory.mkdir(exist_ok=False)
        tree = ET.parse(args.ski)
        root = tree.getroot()
        root.find(".//MonteCarloSimulation").set("numPackets", "8388608")
        root.find(".//Random").set("seed", str(seed))
        for frame in root.findall(".//FrameInstrument"):
            frame.set("numPixelsX", "256")
            frame.set("numPixelsY", "256")
        ski = directory / "seed_check.ski"
        tree.write(ski, encoding="utf-8", xml_declaration=True)
        command = [str(args.binary.resolve()), "-g", "-t", "32", "-i", str(args.ski.resolve().parent),
                   "-o", str(directory.resolve()), str(ski.resolve())]
        with (directory / "console.txt").open("w") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        log = (directory / "seed_check_log.txt").read_text()
        backend = next(line for line in log.splitlines() if "GPU acceleration is" in line)
        if "2 CUDA devices" not in backend:
            raise RuntimeError(f"Expected both GPUs, found {backend}")
        cubes = {path.name: fits.getdata(path).astype(np.float64)
                 for path in directory.glob("*_total.fits")}
        if not cubes or not all(np.isfinite(cube).all() and np.any(cube > 0) for cube in cubes.values()):
            raise RuntimeError(f"Invalid FITS outputs in {directory}")
        images.append(cubes)
        summaries.append({"label": label, "seed": seed, "backend": backend,
                          "timings": [line for line in log.splitlines()
                                      if "Finished primary emission" in line or "Finished simulation" in line]})
        print(f"Completed seed check {label}", flush=True)

    metrics = []
    for name, baseline in images[0].items():
        different = images[1][name]
        repeated = images[2][name]
        denominator = np.linalg.norm(baseline.ravel())
        seed_change = np.linalg.norm((different - baseline).ravel()) / denominator
        repeat_error = np.linalg.norm((repeated - baseline).ravel()) / denominator
        metrics.append({"file": name, "different_seed_relative_l2": float(seed_change),
                        "same_seed_relative_l2": float(repeat_error)})
    report = {"purpose": "Seed validation only; 256-pixel frames and 8388608 packets",
              "runs": summaries, "metrics": metrics,
              "passes": all(m["different_seed_relative_l2"] > 1e-4
                            and m["same_seed_relative_l2"] < 1e-5 for m in metrics)}
    (args.output / "seed_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passes"]:
        raise RuntimeError("Seed sensitivity or repeatability failed")


if __name__ == "__main__":
    main()
