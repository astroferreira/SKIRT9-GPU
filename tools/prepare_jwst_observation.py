#!/usr/bin/env python3
"""Derive a NIRCam/MIRI image and NIRSpec-resolution SED setup from an existing galaxy."""

import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


NIRCAM = ("F070W", "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W")
MIRI = ("F560W", "F770W", "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W")


def replace_child(parent, tag, attributes):
    for child in list(parent):
        parent.remove(child)
    return ET.SubElement(parent, tag, attributes)


def count_rows(path):
    with path.open("rb") as stream:
        return sum(bool(line.strip()) and not line.lstrip().startswith(b"#") for line in stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ski", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redshift", type=float, default=1.0)
    parser.add_argument("--distance-mpc", type=float, default=10.,
                        help="Fiducial observer distance at z=0; cosmological distance is used at z>0")
    parser.add_argument("--packets", type=int, default=10000000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resolution", type=float, default=1000.)
    parser.add_argument("--pixels", type=int, default=1000)
    args = parser.parse_args()
    if not (0 <= args.redshift <= 15 and args.packets > 0 and 0 <= args.seed <= 1000000
            and math.isfinite(args.distance_mpc) and args.distance_mpc > 0
            and math.isfinite(args.resolution) and args.resolution > 0 and args.pixels > 0):
        parser.error("Invalid redshift, distance, packets, seed, resolution, or pixels")
    tree = ET.parse(args.base_ski)
    root = tree.getroot()
    simulation = root.find(".//MonteCarloSimulation")
    simulation.set("simulationMode", "DustEmission")
    simulation.set("userLevel", "Expert")
    simulation.set("numPackets", str(args.packets))
    root.find(".//Random").set("seed", str(args.seed))
    replace_child(root.find(".//cosmology"), "FlatUniverseCosmology", {
        "redshift": str(args.redshift), "reducedHubbleConstant": "0.6774", "matterDensityFraction": "0.3089"})

    # Heating depends on the emitted UV/optical field, even for infrared-only observers.
    source = root.find(".//SourceSystem")
    source.set("minWavelength", "0.01 micron")
    source.set("maxWavelength", "30 micron")
    for bias in source.findall(".//LogWavelengthDistribution"):
        bias.set("minWavelength", "0.01 micron")
        bias.set("maxWavelength", "30 micron")
    for family in source.findall(".//BruzualCharlotSEDFamily"):
        family.set("resolution", "High")

    input_counts = {}
    for element in root.iter():
        filename = element.get("filename")
        if filename:
            path = Path(filename)
            if not path.is_absolute():
                path = args.base_ski.resolve().parent / path
            if not path.is_file():
                raise FileNotFoundError(path)
            element.set("filename", str(path.resolve()))
            input_counts[str(path.resolve())] = count_rows(path)

    medium = root.find(".//MediumSystem")
    for tag in ("radiationFieldOptions", "dustEmissionOptions", "secondaryEmissionOptions"):
        for old in medium.findall(tag):
            medium.remove(old)
    rf = ET.SubElement(ET.SubElement(medium, "radiationFieldOptions", type="RadiationFieldOptions"),
                       "RadiationFieldOptions")
    ET.SubElement(ET.SubElement(rf, "radiationFieldWLG", type="DisjointWavelengthGrid"),
                  "LogWavelengthGrid", minWavelength="0.01 micron", maxWavelength="30 micron", numWavelengths="400")
    dust = ET.SubElement(ET.SubElement(medium, "dustEmissionOptions", type="DustEmissionOptions"),
                         "DustEmissionOptions", dustEmissionType="Stochastic", wavelengthBias="0.5")
    ET.SubElement(ET.SubElement(dust, "dustEmissionWLG", type="DisjointWavelengthGrid"),
                  "LogWavelengthGrid", minWavelength="0.1 micron", maxWavelength="1000 micron", numWavelengths="1000")
    ET.SubElement(ET.SubElement(dust, "wavelengthBiasDistribution", type="WavelengthDistribution"),
                  "LogWavelengthDistribution", minWavelength="0.1 micron", maxWavelength="1000 micron")

    system = root.find(".//InstrumentSystem")
    band_grid = ET.Element("ConfigurableBandWavelengthGrid")
    bands = ET.SubElement(band_grid, "bands", type="Band")
    band_names = [f"JWST_NIRCAM_{band}" for band in NIRCAM] + [f"JWST_MIRI_{band}" for band in MIRI]
    for name in band_names:
        ET.SubElement(bands, "BroadBand", bandName=name)
    default_grid = system.find("defaultWavelengthGrid")
    default_grid.clear()
    default_grid.set("type", "WavelengthGrid")
    default_grid.append(copy.deepcopy(band_grid))
    instruments = system.find("instruments")
    frames = list(instruments.findall("FrameInstrument"))
    if not frames:
        raise ValueError("Base model must contain at least one FrameInstrument")
    for old in list(instruments):
        instruments.remove(old)
    views = []
    for frame in frames:
        view = frame.get("instrumentName")
        views.append(view)
        frame.set("distance", f"{args.distance_mpc:g} Mpc" if args.redshift == 0 else "0 Mpc")
        frame.set("numPixelsX", str(args.pixels))
        frame.set("numPixelsY", str(args.pixels))
        grid = frame.find("wavelengthGrid")
        grid.clear()
        grid.set("type", "WavelengthGrid")
        grid.append(copy.deepcopy(band_grid))
        instruments.append(frame)
        attributes = {key: frame.get(key) for key in ("distance", "inclination", "azimuth", "roll")}
        attributes.update(instrumentName=f"{view}_nirspec", radius="0 pc", recordComponents="false",
                          recordPolarization="false", recordStatistics="false")
        sed = ET.SubElement(instruments, "SEDInstrument", attributes)
        ET.SubElement(ET.SubElement(sed, "wavelengthGrid", type="WavelengthGrid"), "ResolutionWavelengthGrid",
                      minWavelength="0.7 micron", maxWavelength="5.3 micron", resolution=str(2 * args.resolution))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    report = {
        "base_ski": str(args.base_ski.resolve()), "ski": str(args.output.resolve()),
        "redshift": args.redshift, "packets_per_phase": args.packets, "seed": args.seed,
        "fiducial_distance_mpc": args.distance_mpc if args.redshift == 0 else None,
        "distance_convention": "Fiducial local distance" if args.redshift == 0 else "Cosmological distance",
        "views": views, "bands_observer_frame": band_names, "pixels": args.pixels,
        "input_table_rows": input_counts,
        "image_accumulator_bytes_per_host_or_gpu": len(views) * len(band_names) * args.pixels**2 * 8,
        "spectrum": {"observer_range_micron": [0.7, 5.3], "target_resolving_power": args.resolution,
                     "samples_per_resolution_element": 2,
                     "bins": 1 + math.ceil(math.log(5.3 / 0.7) / math.log1p(1 / (2 * args.resolution))),
                     "requires_postprocessing": "Convolve to Gaussian FWHM lambda/R; no instrumental LSF applied yet"},
        "status": "Prepared, not executed or converged",
        "limitations": ["Stellar/MAPPINGS template resolution limits the recoverable spectral structure",
                        "No PSF, detector noise, slit losses, velocity broadening, or disperser gaps applied",
                        "Source-table, radiation-field, dust, geometry, and scratch memory are additional to detector memory"],
        "instrument_sources": [
            "https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters",
            "https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers"],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
