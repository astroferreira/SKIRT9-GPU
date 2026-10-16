# SKIRTGPU Observatory

A local web application for composing broadband images, three-filter RGB composites, and integrated SEDs from exported
AREPO/TNG galaxy catalogues. The browser displays the actual source positions in
3D, shows a live camera preview and physical FOV, and submits observations to the
existing `skirtgpu` executable. Results include downloadable science FITS, display
PNG, model configuration, execution log and provenance. SED observations also
include raw SKIRT spectra, CSV tables and an interactive spectral plot.
The SED plot now includes selectable spectral-line and dust-band references with
physical explanations, using each saved observation's redshift. See
[SED reference features](../Docs/SPECTRAL_FEATURES.md) for scope and sources.

The interactive viewer also offers a browser-GPU **Band light + dust** mode:
SKIRT-template band colours, full-catalogue volume reconstruction, and ray-marched
diffuse extinction. See [GPU preview controls and scientific limits](GPU_PREVIEW.md).

## Start

From this directory:

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.lock
npm ci
npm run build
.venv/bin/python -m observatory --port 8765
```

Open <http://127.0.0.1:8765>. The app binds only to loopback and serves its own
JavaScript bundle; the browser needs no CDN access. On the current machine, the
environment and bundle have already been installed/built.

To serve a trusted network directly, pass `--host 0.0.0.0`. This app has no
authentication; restrict port 8765 at the host and upstream firewalls.

To keep the service running independently of a terminal:

```bash
tmux new-session -d -s skirt-observatory \
  -c /home/astroferrreira/SKIRT/git/visualizer \
  '.venv/bin/python -m observatory --port 8765'
tmux attach -t skirt-observatory
```

The service started during implementation is in `skirt-observatory`. To stop it,
attach and press Ctrl-C. This cancels outstanding jobs, terminates the subprocess
group and retains completed products. Do not start a second server against the
same storage: a file lock enforces one render worker.

For a laptop with an existing SSH alias for this host:

```bash
ssh -N -L 8765:127.0.0.1:8765 papareia
```

Then open the same localhost URL on the laptop. Keep any jump-host configuration
in the existing SSH alias. This app is a local, single-user workspace; it does not
implement public hosting, accounts, or shared-user access control.

## Using the studio

1. Select a galaxy. The initial AREPO snapshot 095 model has 768,270 stellar
   particles, 13,899 MAPPINGS young-source particles and 137,557 input dust sites.
2. Orbit the **3D scene** independently and choose **Use this view** to place the
   observation camera. The green box shows its parallel projection and physical
   FOV. Its finite drawn depth is a visual guide; the real instrument is distant.
3. Switch to **Camera view** for direct composition: drag to orbit, right-drag
   (or Shift-drag) to pan, and scroll to change the physical FOV. Angle sliders,
   XY/XZ/YZ presets, target coordinates and reset are also available.
   To target a companion or off-center feature, click **Pick center** and then
   click its particle concentration in either view. The picker takes the median
   3D location of displayed particles close to the click ray, recenters the
   orbit controls, and updates the observation target used by SKIRT. For raw
   MOOSE data this also becomes the center of the on-demand physical export.
4. Check one or more instruments: **Broadband image**, **RGB image**, and
   **Integrated SED**. Choose the single broadband filter, independently assign
   the red/green/blue filters, and/or configure the spectral mode. All selected
   instruments are written to one SKIRT simulation and run together. Choose the
   photon budget and image dimensions, then start the background observation.
   You can continue composing, queue another view, or cancel a pending/running render.
5. The result displays automatically. The history keeps each observation's
   original settings and supports restoring the view. Identical successful
   requests reuse existing products. Changing settings marks the displayed
   completed image as belonging to an earlier view.

Star/young-source/dust toggles and preview exposure affect only the catalogue
display. They never remove sources or dust from the simulation. Each displayed
stellar source is spread with SKIRT's projected cubic-spline profile out to the
distance of its 32nd nearest other source, measured against the full stellar
catalogue (not just the displayed sample). Stellar colour maps population age
continuously from blue to amber/red, while intensity uses a compressed
initial-mass/age luminosity proxy. Young-source intensity follows SFR. Raw gas
is shown as a brighter purple mass-weighted glow; the brown raw dust layer is a
metal-mass proxy. Uniformly sampled particles are weighted by their represented
full-catalogue count, accumulated additively into a floating-point HDR buffer,
then displayed through one filmic exposure pass. This preserves faint structure
around bright nuclei more like the final rendered products without pretending
to calculate their spectrum. These mappings approximate structure and source
crowding, not band-specific photometry, attenuation, scattering or dust
emission. A square on-screen frame and the inset preview use the same camera
basis as the FITS instrument.

The collapsed **Preview shader controls** mirror the useful display controls in
the companion `galaxy_shaders` ray-tracing view. Stellar kernels default to
`2.0×` their physical 32-neighbour radius and gas/dust kernels to `1.5×` their
catalogue radius. PSF softness uses its separable Gaussian pass; Structure is an
unsharp mask; and Linear, Log, and Asinh select the display stretch before ACES
tone mapping. Reset restores these defaults. None of these settings enters a
SKIRT request or modifies a saved scientific product.

### MOOSE raw snapshots

The **MOOSE raw snapshots** group exposes all 198 matching snapshot/group-catalogue
pairs in `/data/RUNs/output` (snapshots 000 through 197). Each entry is centered
on the primary central subhalo selected by `GroupFirstSub[0]`. The interactive
preview reads particle coordinates and weights directly from the HDF5 snapshot;
it does not require pre-generated `.dat` files. Stars are coloured by their
exact snapshot-cosmology age, gas is a bright purple glow, and the brown dust
layer is explicitly a display-only gas-mass times metallicity proxy because
AREPO does not store a separate dust particle type here.

Submitting any image, RGB, or SED instrument generates the BPASS/TODDLERS and diffuse-dust tables
for a physical cube centered on the current camera target, then builds the SKIRT
configuration and runs the normal worker. **Physical model region** controls the
cube size independently of the image FOV and must be at least as large as that
FOV. A larger cube retains more surrounding sources and dust (and therefore more
external illumination) but uses more memory. Exports are content-addressed by
snapshot fingerprint, physical center, model size, and physics version, so an
identical later request reuses the cached physical tables. Integrated SEDs use
the same selected physical model cube but retain their whole-model observer.

On-demand dust choices are explicit `physical_options` in each raw-series entry:
`diffuse_dust` is `torrey` or `rr2014`, and `diffuse_grain_mix` is `dl07` or
`wd01`. These choices enter both export and observation cache identities.
Absent options retain the historical Torrey/DL07 behavior. The deployed registry
explicitly selects RR2014/DL07 for new raw exports and the eight revised physical
models. Prior Torrey/DL07 entries remain under IDs ending in `-torrey-dl07`.
This retains BPASS/TODDLERS and all mesh generators without whole-cell cloud
subtraction, and keeps the original 0.05-2000 micron source range and grids.
The separate WD01 comparison limits transported wavelengths to 900 micron;
its opacity tables stop at 1000 micron, still short of SKIRT's mandatory
2000 micron Planck cooling integrals. That approximation remains documented.
Combined implicit-cloud and diffuse dust mass remains unknown. Adoption evidence and matched FITS:
`/home/astroferrreira/SKIRT/diagnostics/resolved_dust_20260911/`.

## Models and science conventions

### Resolved source velocities

**Source velocities** is an opt-in observation setting (`source_velocities: true`
in the API). BPASS uses stellar particle velocities; TODDLERS uses star-forming
gas velocities. Separate cached source tables retain the original source
properties and RR2014/Draine-Li dust. The recorded systemic reference is the
mass-weighted stellar velocity in the model region. Diffuse dust stays stationary,
and no unresolved velocity dispersion is added. Existing observations are not
changed. The manifest records the conversion, frame, input hashes and limitations.

The Doppler-aware GPU detector preserves resident transport for supported models.
Physics, implementation boundaries, analytic CPU/GPU checks and matched runtime
evidence are in [SOURCE_KINEMATICS.md](../Docs/SOURCE_KINEMATICS.md).

### TNG100 snapshot 80, galaxy 10604

Select **TNG100 · Galaxy 10604** to view the approved **150 kpc physical cube**
centered on galaxy 10604, drawn from its downloaded parent FoF halo 1. This is
not the full several-megaparsec halo: images and integrated SEDs use only the
exported region, even if the camera FOV is increased. The original halo HDF5
is retained unchanged in the exporter's `particledata` directory.

This model contains 1,221,735 BPASS stellar sources, 91 SFR-normalized TODDLERS
sources and 215,024 gas/Voronoi sites, including zero-dust mesh generators.
The 3D preview samples at most 100,000 points per layer; SKIRT uses the complete
regional catalogues. Both snapshot and observer redshifts are 0.2613432561610123,
with the snapshot's Hubble parameter and matter density retained.

The original export used stochastic Draine–Li dust, the Torrey
ISM prescription and diffuse dust-to-metal ratio 0.2; the model metadata
identifies any later dust-only revision. This mini snapshot lacks
elemental abundances, so gas temperatures use the exporter's hydrogen mass
fraction assumption of 0.75. BPASS metallicities outside its grid are clipped
and audited (210,509 sources; 16.59% of initial stellar mass). Full metadata,
catalogue checksums, preparation checks and the reproducible export scripts are
under `/home/astroferrreira/SKIRT/diagnostics/tng100_80_10604_observatory_20260910`.
Registration validates the catalogue and image/SED configurations; it does not
constitute a completed radiative-transfer run or a convergence test.

### Other datasets and conventions

**TNG100 · Halo 1 · full region** includes every downloaded parent-halo member,
with no spatial crop, in an 8 Mpc centered box. The member positions span about
5.51 Mpc on their longest axis; this is the full FoF halo, not the full simulation
volume. The 150 kpc **TNG100 · Galaxy 10604** model remains separately selectable.
Camera FOV and pan support megaparsec scales (up to 10,000 kpc).

The full halo is **view-only on this host**. Its 27,282,645 Voronoi cells and
500-bin heating grid require about 305 GiB for the three double-precision
radiation-field arrays alone, before mesh, sources, dust and GPU allocations.
The application disables both image and SED submission for this registry entry
and enforces the restriction at the worker API. It does not silently subsample
the RT mesh or change the dust physics. Use the 150 kpc model for observations;
the complete physical full-halo tables are retained for a larger-memory or
separately validated alternative-grid workflow.

For the new NIRSpec/MIRI SED option, see [SED_WORKFLOW.md](SED_WORKFLOW.md).
Check **Integrated SED**, choose a spectral preset, and run the observation.
An SED-only run has exactly one SED observer and no image detector; when image
checkboxes are also selected, the SED observer and frame instruments share the
same SKIRT run. The SED integrates the whole model along the selected viewing
direction; the image FOV, target offsets and roll are not used. The result
includes raw data and a separately identified ideal resolution convolution.


`datasets.json` is an explicit server-side registry of existing `.ski` models.
Its paths resolve relative to that registry. In addition to TNG100 above, the
configured galaxies are AREPO 095; TNG50 snapshot 50 galaxy 257247; TNG50 snapshot
99 galaxy 355724; and TNG50 snapshot 50 galaxy 400782. Their legacy configurations
remain separately selectable. The displayed observer redshift comes from the
model's cosmology and need not equal the simulation snapshot redshift. In
particular, the existing 400782 export uses a fixed 10 Mpc rest-frame observer.

The display adapter supports the exporter's standard pc-coordinate tables:
BC03/BPASS stellar particles, MAPPINGS/TODDLERS particles, and Voronoi/particle dust media.
Custom `useColumns`, other stellar families, or velocity-bearing source tables
require a matching adapter. Unsupported schemas are rejected. Catalogue caching
reads in bounded chunks, retains at most 100,000 points per input table with a
deterministic uniform sample, and records exact input row counts and bounds.
The sampled display is not a density estimator or a selection of SKIRT sources.

The legacy AREPO dust input in `SKIRT/workfolder` is **five columns**,
`x y z h dust_mass`, but its old Voronoi setup reads the fourth column as mass.
This app explicitly converts it into a separate four-column `x y z dust_mass`
table using columns `[0, 1, 2, 4]`. That conversion is declared in the registry
and every render manifest. The old input and historical products are untouched.
The corrected input gives a SKIRT-reported original dust mass of approximately
`1.0016e8 Msun`, so new images are not direct reproductions of the old column-
mismatched AREPO images. Existing sanitized TNG dust tables are used directly.

Each job copies the selected base configuration and changes only the random
seed, photon count, and selected instrument suite, removing diagnostic probes. It keeps
the full particle tables, source wavelength ranges and spectral templates,
medium/geometry, heating grids, dust emission physics and cosmology. Equilibrium
and stochastic models remain explicitly distinguished. The selected instruments,
each with a local wavelength grid, replace the previous suite. Broadband images
use one filter; RGB uses one three-band `FrameInstrument`; and the SED uses its
configured `ResolutionWavelengthGrid`.

Camera angles and projected target offsets follow the actual
[`FrameInstrument.cpp`](../SKIRT/core/FrameInstrument.cpp) equations. Positions
are displayed in kpc and serialized to pc. `right × up = observer` is checked
over 500 orientations against the Three.js orthographic camera. FITS Y increases
upwards, so only the PNG display receives a vertical flip. No FITS pixel is
changed by display processing.

Single-band PNGs use a grayscale asinh display (`strength=30`, white point at the 99.5th
percentile of positive pixels). RGB PNGs independently apply the same mapping to
the selected red, green, and blue FITS planes. The original single-band FITS or
three-plane RGB FITS cube is never altered. Each channel has its own recorded
white point, so
PNG brightness should not be compared quantitatively between observations.
FITS shape, finite values and nonnegative intensities are checked before
publication; an empty FOV produces a blank image and explicit warning.

These are idealized Monte Carlo observations: there is no PSF, detector noise,
or demonstrated noise convergence. Legacy WD01 models report limited dust
opacity coverage; revised DL07 models avoid that inherited cooling-range gap.
SKIRT can remove a few invalid or duplicate Voronoi sites. Warnings remain visible in the render
details and manifest. No performance or CPU/GPU parity claim follows from the
viewer or its smoke tests; the broader SKIRTGPU acceleration target is separate.

## Jobs, provenance and configuration

The API validates filters, camera values, resolution and quality, launches the
executable without a shell and runs at most one SKIRT subprocess at a time.
Five queued/active observations are allowed. Progress uses actual SKIRT log
phases and percentages within the current phase, not an invented overall ETA.
Cancellation first sends SIGTERM to the job's process group, then escalates to
SIGKILL after three seconds if required. The execution limit is two hours for
1M through 100M presets, six hours for 1B, and 24 hours for 10B. Browser refresh
does not cancel jobs. On server restart, unfinished records are marked
interrupted; they are not silently rerun.

Jobs live in `var/jobs/<id>/` and catalogue caches in `var/catalogues/`. Completed
artifacts are retained until explicitly removed. Manifests include the request,
camera basis, input sizes/mtime stamps, the base-model SHA256, adapter mapping,
command, environment overrides, image units/stretch, warnings and GPU activation
evidence. Cache identities include the model/input fingerprint and executable
size/mtime. Input identity uses filesystem metadata, not a full content hash of
multi-GB catalogues; replacing content while preserving both size and timestamp
requires a new storage directory or explicit cache invalidation.

Available environment variables:

| Variable | Default / purpose |
| --- | --- |
| `SKIRT_VIEWER_DATASETS` | Path to `datasets.json` |
| `SKIRT_VIEWER_STORAGE` | `visualizer/var` |
| `SKIRT_VIEWER_BINARY` | `SKIRT/release/SKIRT/main/skirtgpu` |
| `SKIRT_VIEWER_THREADS` | 16 host threads |
| `SKIRT_VIEWER_DUST_BATCH_CELLS` | unset: SKIRTGPU sizes stochastic-dust GPU launches (about 1/16 of the emitting cells); accepts 1-1048576 |

The worker sets `SKIRTGPU_PRECOMPUTE_SOURCE_SPECTRA=0` and
`SKIRTGPU_BATCH_FIRST_FORCED_CHUNK=2097152`, and uses the configured host thread count
for primary feeders. Other inherited `SKIRTGPU*` variables are cleared for predictable
execution. Since the 2026-09-30 throughput pass ([notes](../Docs/THROUGHPUT_20260930.md)),
GPU source launch is left enabled: SKIRTGPU builds per-particle spectra in a bounded
windowed GPU cache, which halved primary emission for the AREPO-095 model. Dust
self-absorption iterations also run on the resident GPU path, and the stochastic dust
kernel needs no per-cell scratch memory. A standard 1e7-packet AREPO-095 render dropped
from 187 s to about 50 s. Exact activation is taken from `observation_log.txt`;
unsupported GPU paths still use SKIRT's CPU fallback. Each job records these settings.

The implementation uses [Three.js orthographic cameras](https://threejs.org/docs/#api/en/cameras/OrthographicCamera)
and a [FastAPI service](https://fastapi.tiangolo.com/), with a dedicated subprocess
worker for long simulations. No frontend framework or separate frontend server
is needed once the bundle is built.

## Verification

```bash
.venv/bin/python -m pytest tests/test_pipeline.py tests/test_spectra.py -q
npm test
npx playwright install chromium
npm run test:browser
```

The browser tests expect a running service on port 8765. Override its URL with
`SKIRT_VIEWER_TEST_URL`. Set `SKIRT_VIEWER_CHROMIUM` to reuse an installed Chromium.
For an actual GPU run through the browser (rather than only UI checks):

```bash
SKIRT_VIEWER_REAL_GPU=1 npm run test:browser
```

That test submits two distinct AREPO observations, checks FITS/image availability
and GPU evidence, reuses an identical request, changes angle/FOV/target/filter,
reloads history and restores an earlier view. It reuses saved matching results
on subsequent runs. Unit tests use a clearly marked fake subprocess only to
exercise queue lifecycle and cancellation; they are not evidence of GPU results.

See [VALIDATION.md](VALIDATION.md) for the actual run measurements and limitations.
For the NIRSpec/MIRI controls, spectral products and opt-in SED-only GPU browser
test, see [SED_WORKFLOW.md](SED_WORKFLOW.md).
