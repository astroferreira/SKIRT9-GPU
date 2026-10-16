"""Camera projection, validated simulation requests and faithful FITS display."""

import copy
import math
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np
from astropy.io import fits
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import table_chunks
from .spectra import SPECTRAL_MODES, MIN_RESOLVING_POWER, MAX_RESOLVING_POWER, expected_bins, resolved_spectral_mode


BANDS = {
    **{f"JWST_NIRCAM_{name}": {"group": "JWST · NIRCam", "label": name, "pivot_micron": pivot}
       for name, pivot in zip(("F070W", "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"),
                              (.7039, .9022, 1.154, 1.501, 1.989, 2.762, 3.568, 4.404))},
    **{f"JWST_MIRI_{name}": {"group": "JWST · MIRI", "label": name, "pivot_micron": pivot}
       for name, pivot in zip(("F560W", "F770W", "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"),
                              (5.635, 7.639, 9.953, 11.31, 12.81, 15.06, 17.98, 20.80, 25.36))},
    **{f"SLOAN_SDSS_{name}": {"group": "Sloan · SDSS", "label": name.lower(), "pivot_micron": pivot}
       for name, pivot in zip("UGRIZ", (.35565, .47024, .61755, .74899, .89467))},
}
QUALITY = {"preview": {"packets": 1000000, "label": "Preview · 1M packets"},
           "standard": {"packets": 10000000, "label": "Standard · 10M packets"},
           "deep": {"packets": 100000000, "label": "Deep · 100M packets"},
           "billion": {"packets": 1000000000, "label": "Ultra · 1B packets", "timeout_multiplier": 3},
           "ten_billion": {"packets": 10000000000, "label": "Extreme · 10B packets", "timeout_multiplier": 12}}


class Camera(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    inclination: float = Field(default=55, ge=0, le=180)
    azimuth: float = Field(default=35, ge=-180, le=180)
    roll: float = Field(default=0, ge=-180, le=180)
    target_kpc: tuple[float, float, float] = (0., 0., 0.)
    fov_kpc: float = Field(default=60, ge=.1, le=10000)

    @field_validator("target_kpc")
    @classmethod
    def target_in_domain(cls, value):
        if any(abs(v) > 10000 for v in value):
            raise ValueError("Camera target must be within 10000 kpc of the model origin")
        return value


class RGBBands(BaseModel):
    model_config = ConfigDict(extra="forbid")
    red: str = "JWST_NIRCAM_F444W"
    green: str = "JWST_NIRCAM_F200W"
    blue: str = "JWST_NIRCAM_F115W"


class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dataset_id: str = Field(min_length=1, max_length=100)
    band: str = "JWST_NIRCAM_F150W"
    camera: Camera = Field(default_factory=Camera)
    quality: Literal["preview", "standard", "deep", "billion", "ten_billion"] = "preview"
    pixels: Literal[256, 512, 1024] = 512
    seed: int = Field(default=12345, ge=0, le=1000000)
    # ``output`` remains the preferred product for backwards compatibility.
    # New clients use ``instruments`` to request several products in one run.
    output: Literal["image", "rgb", "sed"] = "image"
    instruments: tuple[Literal["image", "rgb", "sed"], ...] | None = Field(default=None, min_length=1, max_length=3)
    rgb_bands: RGBBands = Field(default_factory=RGBBands)
    spectral_mode: str = "nirspec_prism"
    spectral_resolution: int | None = Field(default=None, ge=MIN_RESOLVING_POWER, le=MAX_RESOLVING_POWER, strict=True)
    model_fov_kpc: float | None = Field(default=None, ge=.1, le=10000)
    source_velocities: bool = Field(default=False, strict=True)

    @field_validator("spectral_mode")
    @classmethod
    def known_spectral_mode(cls, value):
        if value not in SPECTRAL_MODES:
            raise ValueError("Unknown spectral observing mode")
        return value

    @model_validator(mode="after")
    def valid_instruments_and_bands(self):
        selected = self.selected_instruments()
        if len(selected) != len(set(selected)):
            raise ValueError("Each instrument can be selected only once")
        bands = list(self.rgb_bands.model_dump().values())
        if self.band not in BANDS or any(band not in BANDS for band in bands):
            raise ValueError("Unknown broadband filter")
        if "rgb" in selected and len(set(bands)) != 3:
            raise ValueError("RGB channels must use three distinct broadband filters")
        return self

    def selected_instruments(self):
        return self.instruments or (self.output,)

    def primary_product(self):
        selected = self.selected_instruments()
        if self.output in selected:
            return self.output
        return next((product for product in ("rgb", "image", "sed") if product in selected), selected[0])

    def render_identity(self):
        payload = self.model_dump(mode="json")
        if not self.source_velocities:
            payload.pop('source_velocities')
        selected = self.selected_instruments()
        # Display preference and checkbox order do not change the simulation.
        if self.instruments is None:
            payload.pop("instruments")
        else:
            payload["instruments"] = sorted(selected)
            payload.pop("output")
        if "rgb" not in selected:
            payload.pop("rgb_bands")
        if "image" not in selected:
            payload.pop("band")
        # The effective preset below carries R. Images ignore spectral settings.
        payload.pop("spectral_resolution")
        if payload.get('model_fov_kpc') is None: payload.pop('model_fov_kpc')
        if "sed" not in selected:
            # Preserve cache identities of images created before SED support.
            if self.instruments is None and self.output == "image":
                payload.pop("output")
            payload.pop("spectral_mode")
        else:
            payload["spectral_preset"] = resolved_spectral_mode(self.spectral_mode, self.spectral_resolution)
            if not ({"image", "rgb"} & set(selected)):
                payload.pop("pixels")
                payload["camera"] = {k: payload["camera"][k] for k in ("inclination", "azimuth")}
        return payload


def camera_basis(camera):
    theta, phi, omega = map(math.radians, (camera.inclination, camera.azimuth, camera.roll))
    ct, st, cp, sp, cw, sw = math.cos(theta), math.sin(theta), math.cos(phi), math.sin(phi), math.cos(omega), math.sin(omega)
    # Exact rows used by FrameInstrument::pixelOnDetector(). right x up = observer.
    right = np.array([-cw * sp + sw * cp * ct, cw * cp + sw * sp * ct, -sw * st])
    up = np.array([-sw * sp - cw * cp * ct, sw * cp - cw * sp * ct, cw * st])
    observer = np.array([st * cp, st * sp, ct])
    return right, up, observer


def broadband_grid(bands):
    grid = ET.Element("ConfigurableBandWavelengthGrid")
    children = ET.SubElement(grid, "bands", type="Band")
    for band in bands:
        ET.SubElement(children, "BroadBand", bandName=band)
    return grid


def frame_instrument(instruments, name, distance, request, right, up, bands):
    target_pc = np.array(request.camera.target_kpc) * 1000
    frame = ET.SubElement(instruments, "FrameInstrument", instrumentName=name, distance=distance,
                      inclination=f"{request.camera.inclination:.12g} deg", azimuth=f"{request.camera.azimuth:.12g} deg",
                      roll=f"{request.camera.roll:.12g} deg", fieldOfViewX=f"{request.camera.fov_kpc * 1000:.12g} pc",
                      fieldOfViewY=f"{request.camera.fov_kpc * 1000:.12g} pc", numPixelsX=str(request.pixels), numPixelsY=str(request.pixels),
                      centerX=f"{np.dot(right, target_pc):.12g} pc", centerY=f"{np.dot(up, target_pc):.12g} pc",
                      recordComponents="false", recordPolarization="false", recordStatistics="false")
    ET.SubElement(frame, "wavelengthGrid", type="WavelengthGrid").append(broadband_grid(bands))
    return frame


def prepare(dataset, request, directory, cache):
    selected = request.selected_instruments()
    if any(abs(t) > 10000 for t in request.camera.target_kpc):
        raise ValueError("Camera target must be within 10000 kpc of the model origin")
    tree = dataset.tree()
    root = tree.getroot()
    digest, provenance = dataset.fingerprint()
    adapters = []
    for element in root.iter():
        filename = element.get("filename")
        if not filename:
            continue
        path = (dataset.path.parent / filename).resolve()
        if not path.is_file():
            raise ValueError(f"Missing model input: {path}")
        if element.tag == "VoronoiMeshMedium" and dataset.entry.get("legacy_dust_columns"):
            converted = cache / f"{dataset.id}-{digest[:16]}-dust-skirt9.dat"
            if not converted.is_file():
                temporary = converted.with_suffix(".tmp")
                with temporary.open("w") as stream:
                    for chunk in table_chunks(path, dataset.entry["legacy_dust_columns"]):
                        if (chunk[:, 3] < 0).any():
                            raise ValueError("Negative dust masses in legacy input")
                        np.savetxt(stream, chunk, fmt="%.17g")
                temporary.replace(converted)
            adapters.append({"input": str(path), "output": str(converted),
                             "columns_zero_based": dataset.entry["legacy_dust_columns"]})
            path = converted
        element.set("filename", str(path))
    simulation = root.find(".//MonteCarloSimulation")
    simulation.set("userLevel", "Expert")
    simulation.set("numPackets", str(QUALITY[request.quality]["packets"]))
    root.find(".//Random").set("seed", str(request.seed))
    # Keep source spectra, heating grids, emission physics, cosmology and medium intact.
    # Only the requested instrument replaces the previous camera/filter suite.
    system = root.find(".//InstrumentSystem")
    base_frame = system.find(".//FrameInstrument")
    distance = base_frame.get("distance", "10 Mpc")
    for child in list(system):
        system.remove(child)
    spectral = "sed" in selected
    spatial = bool({"image", "rgb"} & set(selected))
    if spectral:
        mode = resolved_spectral_mode(request.spectral_mode, request.spectral_resolution)
        spectral_grid = ET.Element("ResolutionWavelengthGrid", minWavelength=f"{mode['min_micron']} micron",
                                   maxWavelength=f"{mode['max_micron']} micron",
                                   resolution=str(mode["resolving_power"] * mode["samples_per_resolution_element"]))
        # Explicit spectral units make conversions and the downloadable tables
        # unambiguous; the physical model itself remains unchanged.
        units = simulation.find("units")
        if units is None:
            units = ET.SubElement(simulation, "units", type="Units")
        for child in list(units):
            units.remove(child)
        ET.SubElement(units, "ExtragalacticUnits", wavelengthOutputStyle="Wavelength", fluxOutputStyle="Neutral")
    broadband_bands = []
    if "image" in selected:
        broadband_bands.append(request.band)
    if "rgb" in selected:
        broadband_bands.extend(request.rgb_bands.model_dump().values())
    broadband_bands = sorted(set(broadband_bands), key=lambda band: BANDS[band]["pivot_micron"])
    default_grid = broadband_grid(broadband_bands) if broadband_bands else spectral_grid
    ET.SubElement(system, "defaultWavelengthGrid", type="WavelengthGrid").append(copy.deepcopy(default_grid))
    instruments = ET.SubElement(system, "instruments", type="Instrument")
    right, up, observer = camera_basis(request.camera if spatial else request.camera.model_copy(update={"roll": 0}))
    if "image" in selected:
        frame_instrument(instruments, "view", distance, request, right, up, [request.band])
    if "rgb" in selected:
        frame_instrument(instruments, "rgb", distance, request, right, up, request.rgb_bands.model_dump().values())
    if spectral:
        frame = ET.SubElement(instruments, "SEDInstrument", instrumentName="spectrum", distance=distance,
                              inclination=f"{request.camera.inclination:.12g} deg", azimuth=f"{request.camera.azimuth:.12g} deg",
                              roll="0 deg", radius="0 pc", recordComponents="false",
                              recordPolarization="false", recordStatistics="false")
        ET.SubElement(frame, "wavelengthGrid", type="WavelengthGrid").append(spectral_grid)
    # Diagnostic probes can be expensive and aren't part of the requested observation.
    for container in root.findall(".//probes"):
        for probe in list(container):
            container.remove(probe)
    ET.indent(tree, space="    ")
    tree.write(directory / "observation.ski", encoding="utf-8", xml_declaration=True)
    metadata = {"dataset": dataset.info(), "input_fingerprint": digest, "provenance": provenance,
            "input_adapters": adapters, "request": request.model_dump(mode="json"),
            "instruments": list(selected),
            "packets_per_phase": QUALITY[request.quality]["packets"],
            "projection": {"right": right.tolist(), "up": up.tolist(), "observer": observer.tolist(), "units": "kpc"},
            "limitations": ["Catalogue preview is sampled and is not a radiative-transfer image.",
                            "Monte Carlo noise is not demonstrated converged; no telescope PSF or detector noise is applied.",
                            "PNG uses an asinh display stretch; the original FITS values are preserved."]}
    if "rgb" in selected:
        channel_bands = request.rgb_bands.model_dump()
        sorted_bands = sorted(channel_bands.values(), key=lambda band: BANDS[band]["pivot_micron"])
        metadata["rgb_observation"] = {"channels": channel_bands, "fits_plane_order": sorted_bands,
                                        "source": "Three untouched SKIRT broadband planes composed only for display"}
    if spectral:
        metadata["spectral_observation"] = {"mode_id": request.spectral_mode, **mode, "bins": expected_bins(mode),
                                             "aperture": "Whole model; no FOV, target offset or roll restriction",
                                             "frame": "rest" if metadata["dataset"]["distance_mpc"] > 0 else "observer"}
        sed_display_note = "The raw SED is retained; the plotted curve applies an ideal constant-R Gaussian in log wavelength."
        if spatial:
            metadata["limitations"].append(sed_display_note)
        else:
            metadata["limitations"][-1] = sed_display_note
        metadata["limitations"].extend([mode["approximation"], "Native source-template and dust-emission grid resolution limit recoverable spectral detail."])
    return metadata


def render_fits(path, output, pixels):
    with fits.open(path) as hdus:
        frame = np.asarray(hdus[0].data, dtype=np.float64).squeeze()
        units = hdus[0].header.get("BUNIT", "unknown")
        if frame.shape != (pixels, pixels):
            raise ValueError(f"Unexpected FITS image shape: {frame.shape}")
        if not np.isfinite(frame).all() or (frame < 0).any():
            raise ValueError("FITS image contains invalid or negative pixels")
        positive = frame[frame > 0]
        white = float(np.percentile(positive, 99.5)) if positive.size else 1.
        stretched = np.arcsinh(30 * np.clip(frame / white, 0, 1)) / np.arcsinh(30)
        # FITS Y grows upwards; PNG row zero is at the top. Never flip science pixels.
        Image.fromarray(np.flipud(np.rint(stretched * 255).astype(np.uint8))).save(output)
        return {"shape": list(frame.shape), "units": units, "sum": float(frame.sum()), "max": float(frame.max()),
                "positive_pixels": int(positive.size), "blank": positive.size == 0,
                "display": {"stretch": "asinh", "strength": 30, "white_percentile_positive": 99.5,
                            "white_value": white, "vertical_flip_for_png": True, "colormap": "grayscale"}}


def render_rgb_fits(path, output, pixels, rgb_bands):
    """Create a documented display RGB while leaving the SKIRT FITS cube untouched."""
    with fits.open(path) as hdus:
        cube = np.asarray(hdus[0].data, dtype=np.float64)
        units = hdus[0].header.get("BUNIT", "unknown")
        if cube.shape != (3, pixels, pixels):
            raise ValueError(f"Unexpected RGB FITS cube shape: {cube.shape}")
        if not np.isfinite(cube).all() or (cube < 0).any():
            raise ValueError("RGB FITS cube contains invalid or negative pixels")
        channels = rgb_bands.model_dump() if isinstance(rgb_bands, RGBBands) else dict(rgb_bands)
        plane_order = sorted(channels.values(), key=lambda band: BANDS[band]["pivot_micron"])
        plane_for = {band: index for index, band in enumerate(plane_order)}
        display_channels = []
        metadata = {}
        blank = True
        for color in ("red", "green", "blue"):
            band = channels[color]
            frame = cube[plane_for[band]]
            positive = frame[frame > 0]
            blank &= positive.size == 0
            white = float(np.percentile(positive, 99.5)) if positive.size else 1.
            normalized = np.clip(frame / white, 0, 1)
            display_channels.append(np.arcsinh(30 * normalized) / np.arcsinh(30))
            metadata[color] = {"band": band, "fits_plane": plane_for[band], "sum": float(frame.sum()),
                               "max": float(frame.max()), "positive_pixels": int(positive.size), "white_value": white}
        rgb = np.stack(display_channels, axis=-1)
        Image.fromarray(np.flipud(np.rint(rgb * 255).astype(np.uint8)), mode="RGB").save(output)
        return {"shape": list(cube.shape), "units": units, "blank": blank, "channels": metadata,
                "fits_plane_order": plane_order, "display": {"stretch": "per-channel asinh", "strength": 30,
                "white_percentile_positive": 99.5, "vertical_flip_for_png": True,
                "scientific_fits_preserved": True}}
