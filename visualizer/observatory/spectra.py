"""JWST spectral coverage presets and validated, explicitly idealized SED products."""

import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
from astropy.io import fits


NIRSPEC_DOC = "https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters"
MIRI_LRS_DOC = "https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-low-resolution-spectroscopy"
MIRI_MRS_DOC = "https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-medium-resolution-spectroscopy"


def spectral_mode(label, group, low, high, resolution, source):
    return {"label": label, "group": group, "min_micron": low, "max_micron": high,
            "resolving_power": resolution, "samples_per_resolution_element": 2,
            "source": source, "approximation": "Ideal constant-R Gaussian; no throughput, detector gaps, slit losses or calibrated line-spread function."}


SPECTRAL_MODES = {
    "nirspec_prism": spectral_mode("PRISM / CLEAR", "JWST · NIRSpec", .6, 5.3, 100, NIRSPEC_DOC),
    **{f"nirspec_{grating.lower()}_{band.lower()}": spectral_mode(f"{grating} / {band}", "JWST · NIRSpec", low, high, resolution, NIRSPEC_DOC)
       for grating, band, low, high, resolution in [
           ("G140M", "F070LP", .70, 1.27, 1000), ("G140M", "F100LP", .97, 1.89, 1000),
           ("G235M", "F170LP", 1.66, 3.17, 1000), ("G395M", "F290LP", 2.87, 5.27, 1000),
           ("G140H", "F070LP", .81, 1.27, 2700), ("G140H", "F100LP", .97, 1.89, 2700),
           ("G235H", "F170LP", 1.66, 3.17, 2700), ("G395H", "F290LP", 2.87, 5.27, 2700)]},
    "miri_lrs": spectral_mode("LRS · 5–14 μm", "JWST · MIRI", 5., 14., 100, MIRI_LRS_DOC),
    "miri_mrs": spectral_mode("MRS · 4.9–27.9 μm", "JWST · MIRI", 4.9, 27.9, 3000, MIRI_MRS_DOC),
}

MIN_RESOLVING_POWER = 10
MAX_RESOLVING_POWER = 10000


def reference_redshift(ski_path, frame):
    """Use the saved observation, never the currently selected viewer dataset."""
    if frame == 'rest':
        return 0.
    if frame != 'observer':
        return None
    try:
        tree = ET.parse(ski_path)
        flat = tree.find('.//FlatUniverseCosmology')
        if flat is not None:
            redshift = float(flat.get('redshift', 'nan'))
            return redshift if math.isfinite(redshift) and redshift >= 0 else None
        if tree.find('.//LocalUniverseCosmology') is not None:
            return 0.
    except (OSError, ValueError, ET.ParseError):
        pass
    return None


def resolved_spectral_mode(mode_id, resolving_power=None):
    """Keep preset coverage; optionally override ideal R, not template resolution."""
    mode = dict(SPECTRAL_MODES[mode_id])
    if resolving_power is not None:
        if (isinstance(resolving_power, bool) or not isinstance(resolving_power, int)
                or not MIN_RESOLVING_POWER <= resolving_power <= MAX_RESOLVING_POWER):
            raise ValueError(f'SED resolving power must be an integer from {MIN_RESOLVING_POWER} to {MAX_RESOLVING_POWER}')
        mode['resolving_power'] = resolving_power
    return mode


def expected_bins(mode):
    return 1 + math.ceil(math.log(mode["max_micron"] / mode["min_micron"]) /
                         math.log1p(1 / (mode["resolving_power"] * mode["samples_per_resolution_element"])))


def smooth_spectrum(wavelength, flux, resolving_power):
    """Gaussian smoothing of lambda F_lambda (flux per log interval) on a log grid.

    Matches the existing render_jwst_observation.py convention without importing
    its image rendering dependencies. Edge values are extended; the raw spectrum
    is preserved separately. Resolving power is an approximation, not an LSF.
    """
    if len(wavelength) < 3 or wavelength.shape != flux.shape or resolving_power <= 0:
        raise ValueError("At least three matching samples and positive resolving power are required")
    if not np.isfinite(wavelength).all() or (wavelength <= 0).any() or not np.isfinite(flux).all():
        raise ValueError("Invalid spectral samples")
    steps = np.diff(np.log(wavelength))
    step = float(np.median(steps))
    if step <= 0 or not np.allclose(steps, step, rtol=1e-4, atol=1e-10):
        raise ValueError("Spectrum must have a uniformly increasing logarithmic wavelength grid")
    sigma = 1 / (resolving_power * math.sqrt(8 * math.log(2)) * step)
    radius = max(1, math.ceil(4 * sigma))
    kernel = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(flux, radius, mode="edge"), kernel, mode="valid")


def process_sed(path, directory, mode_id, dataset_name, frame, resolving_power=None):
    """Preserve native SKIRT .dat and produce tabular FITS/CSV/JSON and a plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = resolved_spectral_mode(mode_id, resolving_power)
    header = []
    with path.open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            header.append(line.strip())
    if not any(re.fullmatch(r"# column 1: wavelength; lambda \(micron\)", line) for line in header):
        raise ValueError("SED must contain wavelength in micron in column 1")
    if not any(re.fullmatch(r"# column 2: total flux; lambda\*F_lambda \(W/m2\)", line) for line in header):
        raise ValueError("SED must contain total lambda*F_lambda in W/m2 in column 2")
    data = np.loadtxt(path, ndmin=2)
    if data.shape != (expected_bins(mode), 2):
        raise ValueError(f"Unexpected SED shape {data.shape}; expected ({expected_bins(mode)}, 2)")
    wavelength, raw = data.T
    if not np.isfinite(data).all() or (raw < 0).any():
        raise ValueError("SED contains invalid or negative fluxes")
    if not np.allclose(wavelength[[0, -1]], [mode["min_micron"], mode["max_micron"]], rtol=1e-8, atol=0):
        raise ValueError("SED wavelength range does not match the selected spectral mode")
    convolved = smooth_spectrum(wavelength, raw, mode["resolving_power"])
    metadata = {"mode_id": mode_id, "mode": mode, "samples": len(wavelength), "wavelength_units": "micron",
                "flux_units": "W/m2", "flux_quantity": "lambda*F_lambda", "spectral_frame": frame,
                "aperture": "Whole model; no aperture restriction", "positive_bins": int((raw > 0).sum()),
                "blank": not bool((raw > 0).any()), "smoothing": {"kernel": "Gaussian in log wavelength",
                "resolving_power": mode["resolving_power"], "edge_mode": "nearest", "truncate_sigma": 4,
                "native_template_resolution_preserved": True}, "raw_header": header,
                "reference_redshift": reference_redshift(directory/'observation.ski', frame)}
    payload = {**metadata, "wavelength_micron": wavelength.tolist(), "raw_flux": raw.tolist(), "smoothed_flux": convolved.tolist()}
    (directory / "spectrum.json").write_text(json.dumps(payload, allow_nan=False))
    np.savetxt(directory / "spectrum.csv", np.column_stack((wavelength, raw, convolved)), delimiter=",", fmt="%.12e",
               header="wavelength_micron,raw_lambda_Flambda_W_m2,smoothed_lambda_Flambda_W_m2", comments="")
    primary = fits.PrimaryHDU()
    primary.header["SPECMODE"] = mode_id
    primary.header["SPECFRAM"] = frame
    primary.header["APERTURE"] = "ALL"
    primary.header["RESPOWER"] = (mode["resolving_power"], "Ideal Gaussian resolving power")
    primary.header["LSFMODEL"] = "IDEAL"
    primary.header["HISTORY"] = "Original SKIRT SED retained in observation_spectrum_sed.dat"
    primary.header["HISTORY"] = "No throughput, slit, IFU geometry, detector noise or calibrated LSF."
    for line in header:
        primary.header["HISTORY"] = line.lstrip("# ").encode("ascii", errors="replace").decode("ascii")
    columns = [fits.Column(name="WAVELENGTH", format="D", unit="um", array=wavelength),
               fits.Column(name="LAMBDA_FLAMBDA_RAW", format="D", unit="W m-2", array=raw),
               fits.Column(name="LAMBDA_FLAMBDA_SMOOTHED", format="D", unit="W m-2", array=convolved)]
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns, name="SED")]).writeto(directory / "spectrum.fits")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        try:
            ax.plot(wavelength, raw, color="#aac5ba", lw=.65, alpha=.75, label="Raw SKIRT")
            ax.plot(wavelength, convolved, color="#245f51", lw=1.05, label=f"Ideal R ≈ {mode['resolving_power']:,}")
            ax.set_xscale("log")
            if (convolved > 0).any():
                ax.set_yscale("log", nonpositive="mask")
            ax.set_xlabel(f"{frame.capitalize()} wavelength (μm)")
            ax.set_ylabel("λFλ (W m⁻²)")
            ax.set_xlim(mode["min_micron"], mode["max_micron"])
            ax.grid(alpha=.12, which="both")
            ax.set_title(f"{dataset_name}\n{mode['group']} · {mode['label']}", fontsize=12)
            ax.legend(frameon=False, fontsize=9)
            fig.savefig(directory / "spectrum.png", dpi=180)
        finally:
            plt.close(fig)
    return metadata
