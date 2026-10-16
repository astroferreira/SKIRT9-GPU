# Observatory validation — 2026-09-10

The application is running on papareia at `127.0.0.1:8765`, in the tmux session
`skirt-observatory`. It uses the existing executable
`/home/astroferrreira/SKIRT/release/SKIRT/main/skirtgpu`, which reports SKIRT v9.0,
git `2cd3a57-dirty`, built 2026-09-09 at 15:24:36. This work adds the `visualizer/`
application; it does not edit the existing SKIRT C++ changes or original input
catalogues.

## Real observations

Both observations used the full AREPO 095 catalogues, equilibrium dust emission,
observer redshift 1, 1,000,000 primary and secondary photon packets, seed 12345,
16 host threads and one 256 × 256 broadband FrameInstrument. The worker retained
the original source spectra and heating/emission settings, with the documented
legacy dust-column correction.

| Observation | F150W / XY | F277W / XZ, panned |
| --- | --- | --- |
| Job ID | `5ee58388a3984032a8a457f7acb7c51f` | `429286f3d4694bfcbeb9ac605fc6c576` |
| Inclination / azimuth / roll | 0° / 0° / 90° | 90° / −90° / 0° |
| Target (kpc) | (0, 0, 0) | (3, 0, 0) |
| Physical FOV | 60.005814 kpc | 36 kpc |
| End-to-end job wall time | 41.4 s | 40.6 s |
| FITS shape | (1, 256, 256) | (1, 256, 256) |
| Nonzero pixels | 17,091 | 24,171 |
| Pixel sum, native surface-brightness units | 2.521419203e−14 | 5.494583756e−14 |
| GPU activation | Both RTX 5090 devices confirmed | Both RTX 5090 devices confirmed |

FITS units are `W/m2/arcsec2`; a sum of surface-brightness pixels is not an
integrated flux unless multiplied by the solid angle per pixel. Both images
contain only finite, nonnegative values. The panned observation's FITS coordinate
centre is shifted by the requested physical offset. The actual PNG/FITS pairs
were inspected visually and through headers/arrays. The camera preview and FITS
display have matching orientation and framing, with expected differences from
dust radiative transfer, population colouring, point sampling and Monte Carlo
noise.

Each job directory under `var/jobs/` contains its original science FITS,
`observation.ski`, `manifest.json`, `observation_log.txt`, `console.log` and
`preview.png`. The first render's GPU-status metadata was corrected by reading
the authoritative simulation log; the already completed FITS was reused.
The final browser regression reused both completed results, requiring no
additional photon simulation.

## Catalogue checks

All four registered datasets were read completely and selected successfully in
the browser. These are input-table row counts, including zero-dust sites and
sites which SKIRT may subsequently reject when constructing the mesh.

| Dataset | Stellar particles | MAPPINGS particles | Dust sites |
| --- | ---: | ---: | ---: |
| AREPO 095 | 768,270 | 13,899 | 137,557 |
| TNG50 257247 | 1,127,243 | 35,119 | 1,151,007 |
| TNG50 355724 | 5,306,823 | 298,941 | 1,906,618 |
| TNG50 400782 | 101,003 | 13,319 | 177,634 |

The browser retains at most 100,000 display points per table. Cached display
products and the two observations occupy approximately 32 MB; this excludes
the existing full input catalogues, environments and npm dependencies. All 22
configured filter transmission resources were confirmed present in the installed
SKIRT resource packs. Only the two listed NIRCam bands were newly rendered in
this validation; loading a TNG catalogue does not establish a new TNG GPU render.

## Automated checks

- 20 Python tests passed: catalogue sampling/counts/cache invalidation, physical
  camera projection and offsets, preserving source/dust/cosmology configuration,
  dust mass-column conversion, input validation, FITS display orientation and
  unchanged pixels, subprocess completion/cache/restart, queued/running
  cancellation, execution timeout, API request boundaries and actual-log GPU
  activation parsing.
- 2 JavaScript tests passed, including detector projection and handedness for
  500 angle combinations and the XY axis convention.
- 3 browser tests passed: desktop/mobile controls, orbit/pan/FOV/layers, every
  configured dataset, real render submission/results, saved-request reuse,
  changed camera/filter/target, history persistence and restoring an earlier view.
- The existing `skirtgpu -G` backend self-test passed on both RTX 5090s.
- Frontend bundle built successfully. Desktop/mobile screenshots were inspected;
  the mobile test verifies no horizontal page overflow. Screenshots are in
  `test-results/observatory-desktop.png`, `observatory-mobile.png`, and
  `observatory-real-render.png`.

The Python suite emits two upstream deprecation warnings from Starlette's
TestClient/httpx integration; it has no test failures.

## Scientific and operational limits

The existing AREPO configuration still warns that its dust optical-property
resources do not cover the full requested emission wavelength range, and it
removes seven invalid Voronoi sites during mesh construction. These warnings are
retained in the interface and manifests. The previews are visibly noisy at this
photon count; neither noise convergence nor CPU/GPU science parity has been
established here. There is no telescope PSF or detector noise model.

The measurements above are observed latencies for this small preview workload,
not a speedup comparison or evidence of the broader SKIRTGPU performance target.
Large catalogues and higher photon budgets can take substantially longer. The
worker starts a fresh SKIRT process for a new observation; it does not persist
the initialized radiative-transfer model on the GPU between different views.
