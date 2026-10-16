#!/usr/bin/env python3
"""Render completed SKIRT JWST cubes without changing the science outputs."""

import argparse
import html
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d


CHANNELS = {
    "nircam": ("F444W", "F277W", "F115W"),
    "miri": ("F2100W", "F1500W", "F770W"),
}


def smooth_spectrum(wavelength, flux, resolution):
    """Apply a constant-R Gaussian in log wavelength to flux per log interval."""
    if resolution <= 0 or len(wavelength) < 3:
        raise ValueError("Positive resolving power and at least three samples required")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("Wavelengths must be positive and finite")
    if flux.shape != wavelength.shape or not np.all(np.isfinite(flux)):
        raise ValueError("Flux must be finite and match the wavelength array")
    delta = np.diff(np.log(wavelength))
    step = np.median(delta)
    if step <= 0 or not np.allclose(delta, step, rtol=0.03, atol=0):
        raise ValueError("Spectrum must have a uniformly increasing log-wavelength grid")
    sigma = 1.0 / (resolution * np.sqrt(8 * np.log(2)) * step)
    return gaussian_filter1d(flux, sigma, mode="nearest")


def load_renderer(path):
    spec = importlib.util.spec_from_file_location("skirt_rgb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=1000)
    parser.add_argument("--stretch", type=float, default=30)
    parser.add_argument("--label", help="Galaxy identification for the gallery; defaults to the run name")
    args = parser.parse_args()
    if args.stretch <= 0 or args.resolution <= 0:
        parser.error("Stretch and resolution must be positive")
    prefix = args.prefix.resolve()
    label = args.label or prefix.name
    log = prefix.with_name(prefix.name + "_log.txt").read_text()
    if f"Finished simulation {prefix.name} " not in log:
        raise RuntimeError("Simulation has not finished; refusing to render partial outputs")
    renderer = load_renderer(args.renderer)
    parameters = prefix.with_name(prefix.name + "_parameters.xml")
    root = ET.parse(parameters).getroot()
    units = root.find(".//ExtragalacticUnits")
    if units is None or units.get("fluxOutputStyle") != "Neutral":
        raise ValueError("This renderer expects neutral (lambda F_lambda) flux units")
    if units.get("wavelengthOutputStyle") != "Wavelength":
        raise ValueError("This renderer expects wavelength-ordered output")
    redshift = float(root.find(".//FlatUniverseCosmology").get("redshift"))
    frames = root.findall(".//FrameInstrument")
    views = [frame.get("instrumentName") for frame in frames]
    bands_by_view = renderer.load_frame_instrument_band_orders(parameters)
    output = prefix.parent / "products"
    output.mkdir(exist_ok=True)
    products = {"simulation_prefix": str(prefix), "redshift": redshift,
                "label": label,
                "instrument_distances": {frame.get("instrumentName"): frame.get("distance") for frame in frames},
                "packets_per_phase": float(root.find(".//MonteCarloSimulation").get("numPackets")),
                "status": "First-pass products; Monte Carlo noise convergence not established",
                "processing": {"asinh_stretch": args.stretch, "channel_balance_strength": 0.2,
                               "intensity_percentiles": [0.5, 99.8], "psf": "none",
                               "instrument_noise": "none", "rgb_channels": CHANNELS,
                               "target_spectral_resolution": args.resolution,
                               "spectrum_kernel": "Ideal constant-R Gaussian, not a NIRSpec instrument LSF",
                               "template_resolution": "Native template limitations retained"},
                "views": {}}
    products["run_warnings"] = sorted({line.split("! ", 1)[1].strip()
                                        for line in log.splitlines() if "! " in line
                                        and "following warning messages" not in line})
    selected = {name: {} for name in CHANNELS}
    for frame, view in zip(frames, views):
        path = prefix.with_name(f"{prefix.name}_{view}_total.fits")
        bands = [renderer.canonical_band_name(b) for b in bands_by_view[view]]
        with fits.open(path, memmap=True) as hdus:
            cube = np.asarray(hdus[0].data)
            expected = (len(bands), int(frame.get("numPixelsY")), int(frame.get("numPixelsX")))
            if cube.shape != expected or not np.all(np.isfinite(cube)) or np.any(cube < 0):
                raise ValueError(f"Invalid FITS cube: {path}")
            sums = cube.sum(axis=(1, 2), dtype=np.float64)
            if np.any(sums <= 0):
                raise ValueError(f"Empty band in {path}")
            # These 17 filters are supplied in the same ascending pivot order used by SKIRT.
            wavelengths = np.asarray(hdus[1].data.field(0)).reshape(-1)
            approximate = np.array([float(b[1:-1]) / 100 for b in bands])
            if wavelengths.shape != approximate.shape or not np.allclose(wavelengths, approximate, rtol=0.08):
                raise ValueError(f"FITS wavelength axis does not match the configured filter order: {path}")
            products["views"][view] = {
                "fits": str(path), "shape": list(cube.shape), "units": hdus[0].header.get("BUNIT"),
                "field_kpc": [float(frame.get("fieldOfViewX").split()[0]) / 1000,
                              float(frame.get("fieldOfViewY").split()[0]) / 1000],
                "band_pivot_micron": dict(zip(bands, wavelengths.tolist())),
                "band_pixel_sums": dict(zip(bands, sums.tolist())),
            }
            for name, channels in CHANNELS.items():
                selected[name][view] = {
                    color: np.array(cube[bands.index(band)], dtype=np.float32)
                    for color, band in zip("RGB", channels)
                }
            fig, axes = plt.subplots(4, 5, figsize=(13, 10), facecolor="white")
            for index, ax in enumerate(axes.flat):
                ax.set_axis_off()
                if index >= len(bands):
                    continue
                plane = cube[index]
                high = np.percentile(plane[plane > 0], 99.8)
                display = np.arcsinh(args.stretch * plane / high) / np.arcsinh(args.stretch)
                ax.imshow(display, origin="upper", cmap="gray", vmin=0, vmax=1)
                ax.set_title(bands[index], fontsize=10)
            fig.suptitle(f"z={redshift:g} | {view} | 150 kpc | bands scaled individually")
            fig.tight_layout()
            fig.savefig(output / f"{view}_all_bands.png", dpi=130)
            plt.close(fig)

    for name, channels in CHANNELS.items():
        weights = renderer.compute_channel_weights(
            {color: {view: selected[name][view][color] for view in views} for color in "RGB"}, 0.2)
        products["processing"][f"{name}_channel_weights"] = weights
        paths = []
        for view in views:
            rgb = renderer.make_lupton_rgb(selected[name][view], weights, 0.5, 99.8, args.stretch)
            path = output / f"{view}_{name}_rgb.png"
            renderer.save_png(rgb, path)
            paths.append(path)
        renderer.build_montage(paths, [f"{v}: {' / '.join(channels)}" for v in views],
                               output / f"{name}_montage.png")

    overview = Image.new("RGB", (1040, 560 * len(views)), "white")
    labels = ImageDraw.Draw(overview)
    for row, view in enumerate(views):
        for column, name in enumerate(CHANNELS):
            with Image.open(output / f"{view}_{name}_rgb.png") as source:
                thumb = source.resize((500, 500), Image.Resampling.LANCZOS)
                overview.paste(thumb, (column * 520, row * 560 + 35))
            labels.text((column * 520 + 8, row * 560 + 12),
                        f"{view} | {name.upper()} | {' / '.join(CHANNELS[name])}", fill="black")
    overview.save(output / "jwst_overview.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    for view in views:
        path = prefix.with_name(f"{prefix.name}_{view}_nirspec_sed.dat")
        values = np.loadtxt(path)
        wavelength, neutral = values[:, 0], values[:, 1]
        with path.open() as stream:
            headers = [next(stream) for _ in range(3)]
        if "W/m2" not in "".join(headers):
            raise ValueError(f"Unexpected spectrum flux units: {path}")
        broadened = smooth_spectrum(wavelength, neutral, args.resolution)
        # lambda F_lambda = nu F_nu; 1 microJy = 1e-32 W/m2/Hz.
        conversion = wavelength * 1e-6 / 299792458.0 / 1e-32
        raw, smoothed = neutral * conversion, broadened * conversion
        np.savetxt(output / f"{view}_spectrum_R{args.resolution:g}.dat",
                   np.column_stack((wavelength, raw, smoothed)),
                   header="observer_wavelength_micron raw_Fnu_microJy smoothed_Fnu_microJy")
        ax.plot(wavelength, smoothed, lw=0.8, label=view)
        products["views"][view]["spectrum_bins"] = len(wavelength)
    ax.set(xlabel="Observer wavelength (micron)", ylabel="Flux density (microJy)",
           title=f"z={redshift:g} | integrated spectra | ideal Gaussian R={args.resolution:g}", yscale="log")
    ax.legend(frameon=False, ncols=len(views))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "spectra.png", dpi=170)
    plt.close(fig)
    (output / "products.json").write_text(json.dumps(products, indent=2) + "\n")

    sections = []
    for view in views:
        sections.append(f'<section><h2>{html.escape(view)}</h2><div class="images">' + "".join(
            f'<figure><a href="{view}_{name}_rgb.png"><img src="{view}_{name}_rgb.png" '
            f'alt="{view} {name} RGB"></a><figcaption>{name.upper()}: '
            f'R/G/B = {" / ".join(CHANNELS[name])}</figcaption></figure>' for name in CHANNELS) +
            f'</div><p><a href="{view}_all_bands.png">All 17 bands</a> | '
            f'<a href="../{prefix.name}_{view}_total.fits">Science FITS cube</a> | '
            f'<a href="{view}_spectrum_R{args.resolution:g}.dat">Spectrum data</a></p></section>')
    warning_items = "".join(f"<li>{html.escape(warning)}</li>" for warning in products["run_warnings"])
    warning_section = f"<section><h2>Run warnings</h2><ul>{warning_items}</ul></section>" if warning_items else ""
    first_view = products["views"][views[0]]
    ny, nx = first_view["shape"][1:]
    field = first_view["field_kpc"][0]
    packets = products["packets_per_phase"]
    distance_note = (f"Fiducial observer distance: {html.escape(frames[0].get('distance'))}. "
                     "This is not a single JWST pointing.<br>" if redshift == 0 else "")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TNG50 z={redshift:g}: JWST products</title>
<style>body{{font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1200px;color:#222}}
h1{{font-size:26px}}h2{{font-size:20px}}p{{line-height:1.5}}a{{color:#146d63}}
section{{padding:20px 0;border-top:1px solid #ddd}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
figure{{margin:0;min-width:0}}img{{width:100%;height:auto}}figcaption{{font-size:14px}}
@media(max-width:650px){{.images{{grid-template-columns:1fr}}}}</style>
<h1>{html.escape(label)} at z={redshift:g}</h1>
<p>{field:g} kpc field; {nx} x {ny} pixels per view; {packets:g} packets per phase.
{distance_note}
First pass: Monte Carlo noise convergence is not established.</p>
<p>Asinh stretch {args.stretch:g}; no telescope PSF or detector noise applied.
RGBs are false-colour visualizations. Science FITS files are unchanged.</p>
{''.join(sections)}<section><h2>Integrated spectrum</h2>
<a href="spectra.png"><img src="spectra.png" alt="Integrated spectra of four views"></a>
<p>Ideal Gaussian R={args.resolution:g}; not an instrument-specific NIRSpec LSF.
Native stellar and MAPPINGS template resolution limits remain.</p></section>
{warning_section}
<p><a href="products.json">Product provenance and validation</a> |
<a href="../{prefix.name}_parameters.xml">SKIRT parameters</a> |
<a href="../{prefix.name}_log.txt">Run log</a></p></html>'''
    (output / "index.html").write_text(page)
    print(output / "index.html")


if __name__ == "__main__":
    main()
