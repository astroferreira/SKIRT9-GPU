# Browser and backend GPU band preview

Choose **GPU preview → Browser GPU** or **Backend GPU · band ray marching** in the left panel.
The first preparation reads the full catalogues, computes broadband luminosities,
and caches a volume. Browser mode integrates rays in the viewing browser's WebGL2
context. Backend mode keeps the volume in a persistent NVIDIA EGL worker and
returns PNG image frames; no volume binaries are downloaded by that mode.
Neither mode submits a SKIRT job. The existing Python API still does bounded CPU
data preparation and caching; moving rendering does not move that preparation to CUDA.

## Backend operation and access

- The worker is a separate spawned process owned by the Observatory service. It
  starts on demand, survives camera changes, and is stopped with the service.
  `SKIRT_VIEWER_PREVIEW_GPU=1` selects NVIDIA/CUDA device 1 (the default on this
  two-GPU host). EGL enumeration is explicitly mapped through `EGL_CUDA_DEVICE_NV`;
  here CUDA GPU 1 is EGL device 0. A conflicting `GLCONTEXT_DEVICE_INDEX` is rejected.
  The worker checks the NVIDIA vendor and rejects software renderers.
- Both views use one context and share their 3D textures: 512³ is 4 GiB resident
  rather than two 4-GiB copies. Uploads use memory-mapped 16-plane slabs, avoiding
  a single 2-GiB upload per texture. The default LRU texture budget is 6144 MiB,
  configurable with `SKIRT_VIEWER_PREVIEW_CACHE_MIB`; history/output buffers are
  additional. Idle textures and view buffers expire after 120 seconds.
- A server-generated access key is saved at `var/backend-preview.key` with mode
  0600. External clients enter it in the backend access-key field. Use HTTPS, a
  VPN, or an SSH tunnel: plain HTTP does not protect this credential in transit.
  Direct loopback requests with a loopback Host do not need the master key.
  This authentication protects the new GPU endpoints, **not the entire existing
  Observatory**, which retains its previous access policy.
- Session creation returns an independent capability token. Tokens are sent in
  Authorization headers, not URLs or logs, and are held only in page memory.
  There are at most four sessions, with two-minute inactivity expiry and automatic
  reconnection on the next interaction. Leaving backend mode releases its session.
  Worker-side cleanup also handles abandoned/disconnected clients.
- Only one GPU operation is admitted at a time; busy clients retry their latest
  state rather than enqueue frames. Browser revision checks discard obsolete
  camera responses. A worker startup has a 30-second deadline; a frame/load has
  a 60-second deadline, after which only the renderer process is restarted.
- Camera transforms, band-volume keys, display controls and output dimensions are
  validated. Image requests are capped at four megapixels combined and 2048 per
  dimension; the UI limits its largest dimension to 1920. During interaction it
  uses half-sized images and 96 ray steps, then refines to the selected quality
  and eight samples while stationary. The browser retains lightweight camera,
  picking and guide rendering, but does not run volume ray marching in backend mode.
- This version sends lossless PNG frames through bounded HTTP requests. It does
  not yet use NVENC/WebRTC. Timing in the UI is **render + readback + PNG encoding**,
  excluding network latency; it is not a GPU-only timer or an FPS guarantee.
  On backend failure the UI reports the error and restores diagnostic mode;
  browser ray marching remains an explicit fallback, avoiding an automatic 4-GiB
  download. Server-side source deposition, sparse/adaptive grids, scattering and
  SKIRT production transport are separate work.

API endpoints: `GET /api/backend-preview/status`, `POST /api/backend-preview/sessions`,
`POST /api/backend-preview/frame`, and `DELETE /api/backend-preview/sessions/{id}`.
Frames contain a little-endian uint32 JSON-header length, that JSON header, then
the main/mini PNG payloads with lengths and timing information in the header.
The status endpoint also reports whether CPU volume preparation is in progress.

## Controls

- The preview has independent red, green and blue filters. **Use observation RGB
  filters** copies the instrument choices; **Build / update** applies them.
- **Diffuse dust extinction** and **Dust strength** change only the preview.
  They are independent of the diagnostic dust overlay and never disable dust
  in a SKIRT simulation. 1x uses the input dust masses and the stated opacity.
- **Ray quality & volume region** contains 64/96/128/192/256/512 cubed grids, cube width,
  kernel support, 96/192/384 ray steps, and half/75%/full pixel resolution.
  Grid/width/kernel changes require Build / update. Ray and pixel settings are live.
  The defaults are a 256³ grid and 0.5× support; support can be reduced
  to 0.1×. Diagnostic stellar support defaults to 1× (formerly 2×). These are
  display choices, not changes to the physical SKIRT source catalogue.
- The live grid hint shows voxel width and texture/download memory before a
  rebuild. Across a 60 kpc cube, 128³ / 192³ / 256³ yield 468.75 / 312.5 /
  234.375 pc voxels and 64 / 216 / 512 MiB of float32 volume data. The main and
  miniature views have separate WebGL contexts, each requiring its own texture
  copy: 256³ uses about 1 GiB of GPU volume textures, plus browser-side data and
  render buffers. Preparation also needs extra server RAM and time. Lower grids
  remain available for memory-limited clients. The opt-in **512³ · ultra / high
  memory** option gives 117.1875 pc voxels at 60 kpc. It needs 4 GiB of downloads
  and about 8 GiB of GPU textures for these two views alone, plus browser-side
  arrays and server reconstruction working memory; it is not the default.
  On Linux the preparation checks a conservative working-memory estimate against
  available server RAM before each padded deposition/convolution, reserving 25%
  headroom. An oversized reconstruction reports an error instead of attempting
  that allocation. This is a best-effort guard, not a guarantee against changing
  memory pressure. Browser 3D-texture dimension limits are checked before downloads;
  GPU allocation capacity is device-dependent and cannot be guaranteed in advance.
  Backend mode instead shows shared server texture memory; it sends only image frames.
- Build / update centres the volume on the current observation-camera target.
  Orbiting does not rebuild it. Panning outside its bounds requires a rebuild;
  shrinking the cube increases spatial resolution but excludes outer material.
- Eight subpixel/depth samples accumulate in linear radiance while stationary.
  Camera or transport-control changes reset the accumulation.
- Stars and young-source visibility apply in band mode. Age colours, purple gas,
  the dust overlay and the original kernel sliders belong to diagnostic mode.
  Both modes share exposure, optional PSF softness, structure and stretch.
- Exposure now extends down to a logarithmic slider value of -2 (display gain
  2e-7), about 10,000 times fainter than the former minimum. The default slider
  value of 0.38 and its appearance are unchanged. This affects diagnostic,
  browser-volume and backend previews, not source luminosities or SKIRT outputs.

Diagnostic particles now use instanced camera-facing quads with inverse-area
light normalization and screen-space edge filtering. Their kernels are no longer
limited by the hardware point-size cap. This is filtered cubic-spline splatting,
not an implementation of anisotropic EWA or learned Gaussian splatting.

## What the band image represents

Prepared BPASS/BC03 stellar sources and SFR-normalized TODDLERS sources use the
installed SKIRT stored tables and native band transmission curves. Templates are
integrated at population-grid nodes, then interpolated using the table's axis
scales. Source mass/SFR weights are retained. Colour is neutral band luminosity
(`lambda_pivot * mean L_lambda`), consistent with the Observatory's neutral FITS
flux convention, with one common RGB display normalization. Observer bands are
evaluated at the configured observer redshift; fixed-distance views use rest
wavelengths. This interpolation shortcut is not numerically identical to
integrating an individually interpolated SED for every source.

Stellar support radii query the 32 nearest other sources in the full stellar
catalogue. Raw snapshots use all valid stars in a 1.5x padded preview region.
Raw BPASS/TODDLERS source accounting, ages and diffuse dust prescriptions reuse
the installed physical exporter without running an export or writing input files.
Prepared young-source radii retain their catalogue values.

Full-catalogue luminosity and dust mass are deposited with cloud-in-cell weights,
then reconstructed using cubic-spline kernels binned in sqrt(2) support steps.
Unresolved kernels are limited by voxel resolution. Large-kernel padding is
bounded to one additional cube width on each side. Values outside the resulting
support/cropped region are not silently rescaled back into the cube. This is a
multiscale reconstruction on a regular grid, **not an adaptive octree** or exact
Voronoi-cell reconstruction.

Each browser ray integrates emissivity with wavelength-dependent extinction.
The analytic constant-segment solution includes self-attenuation within each
step and remains well defined when dust is zero. Extinction uses the installed
`MeanInterstellarOpticalProps` curve, averaged through each filter and converted
from cross-section per H to cross-section per dust mass. Its two out-of-order
EUV edge knots are sorted before interpolating the 1D curve.

## Scientific limits

- The mean Milky Way curve is **not** the model's configured multigrain opacity.
  Band-averaged opacity also approximates wavelength-resolved broadband transfer.
- Scattering into the line of sight and diffuse heated-dust/PAH emission are not
  implemented. TODDLERS' intrinsic cloud emission is retained when the selected
  template contains it; that does not supply diffuse dust emission for MIRI.
- The regular grid smooths narrow dust lanes and compact sources. Increasing ray
  steps cannot recover detail missing from the volume grid.
- The preview uses common RGB normalization, whereas existing SKIRT RGB PNGs use
  per-channel white points. Compare FITS channels with a controlled shared display
  treatment before attributing colour differences to the transfer calculation.
- No calibrated flux, numerical SKIRT parity, native-pixel convergence, or FPS
  guarantee is claimed. Browser-mode GPU resources belong to the **client**;
  backend-mode volume/render resources belong to the selected server GPU.
- Legacy MAPPINGS and very large view-only halo datasets retain diagnostic mode
  with an explicit unsupported-volume message.

Prepared source arrays and volume files live under
`var/catalogues/browser-volumes/` (or the configured storage equivalent).
Cache identities include input fingerprints, preview settings, resource stamps
and, for raw inputs, the physical-exporter source stamp. Completed products are
published after the binary files are written; only one uncached build runs at a
time. Original catalogues, simulation configurations, saved FITS and jobs remain
untouched.

## Validation

```bash
.venv/bin/python -m pytest -q
npm test
npm run build
SKIRT_PREVIEW_REAL_DATA=1 npx playwright test tests/browser/volume.spec.js
SKIRT_BACKEND_GPU_TESTS=1 .venv/bin/python -m pytest -q tests/test_backend_preview.py
SKIRT_BACKEND_REAL_DATA=1 npx playwright test tests/browser/backend-preview.spec.js
.venv/bin/python tests/benchmark_backend_preview.py EXISTING_VOLUME_CACHE_KEY
PYTHONPATH=. .venv/bin/python tests/compare_preview.py EXISTING_RGB_JOB_ID
```

Tests cover table layout, interpolation, filter/redshift changes, deposition
ordering, kernel luminosity conservation, out-of-cube kernel contributions,
query validation, analytic emission/extinction, and browser-rendered dust/zero-dust
comparisons. The browser suite uses Chromium's SwiftShader by default: it validates
the actual WebGL shader path, not NVIDIA hardware performance. The comparison
script is a CPU reference integration against an existing FITS, with matching
camera/bands and provenance checks; its per-channel normalized morphology scores
are not a radiative-transfer parity test.

Local validation on 2026-09-13: 93 Python tests and 8 JavaScript tests passed.
Nine targeted browser checks passed across camera/layer controls, raw snapshots,
multi-instrument submission, SED controls, synthetic volume transfer and a real
AREPO-095 volume. The optional test that launches new SKIRT SED simulations was
skipped. The live service remained in `skirt-observatory` on `0.0.0.0:8765`, with
59 retained observations and no new simulation jobs submitted by this work.
The saved AREPO-095 RGB job `622f4577dced456eaa324717beb889f8` has matching current
input provenance. At the same camera and F115W/F090W/F070W bands, the 128³ volume
at 0.46875 kpc/voxel gave per-channel linear morphology correlations of
0.894 / 0.915 / 0.936 against its FITS downsampled to 128². Per-channel
total-flux-normalized L1 differences were 0.279 / 0.289 / 0.283: agreement is
useful but visibly imperfect, and these scores do not measure photometric parity.
The source FITS SHA256 was
`f31588a451a04e537a93c0a9768a0085ea2e0847a35bdc7e0ea7b469ac1216e0`.

Resolution-option update on 2026-09-13: 104 Python tests and 9 JavaScript tests
passed. An actual 512³ single-channel synthetic-source reconstruction preserved
unit luminosity to 2.2e-8 relative error and measured the expected 117.1875 pc
voxel width at 60 kpc. This is a grid-reconstruction check, not a full-galaxy
512³ GPU rendering or performance benchmark. Browser control/transport tests use
small synthetic textures while checking submission of the larger grid choices.

Backend validation on 2026-09-14: 112 Python tests passed with the NVIDIA EGL
tests enabled, including analytic linear-radiance checks, dust controls, the
spawned worker/API path, authentication/session limits, texture memory guards
and cleanup. Fourteen JavaScript tests passed, including the lower exposure floor, expired-session
reconnection and obsolete-frame rejection. The real-galaxy browser/backend
comparison measured mean absolute displayed-channel difference 0.0310/255 and
zero browser volume downloads in backend mode; browser fallback tests also passed.

An existing full-catalogue TNG50 PMCTRL001 snapshot-097 512³ volume was rendered
through the live API on NVIDIA GPU 1 (EGL device 0), confirmed in `nvidia-smi`.
Both main (512×512) and miniature (128×128) views reached eight samples at 192
ray steps, sharing 4096 MiB of texture data. Initial upload took 1404 ms; the
last incremental refinement frame took 14.0 ms for render/readback/PNG encoding
and 15.5 ms including localhost HTTP transport. These are measurements of this
view, not an external-network frame-rate guarantee or an eight-sample-per-frame
benchmark. No new volume reconstruction or SKIRT job was required for this test.
The report and actual PNGs are retained in
`var/backend-validation/render-eeg21dnw/` (source volume key
`9fb749f288459b9108583e0b14f49198dced60aeedba35e9dae2fb31cf7b3f27`).
All 60 saved observations were retained during the benchmark.
The main/mini displayed means were 60.05/60.23 out of 255 for the same camera.
Regression tests cover differing framebuffer sizes and cache eviction after
session cleanup. The earlier `render-bl0i57ib` prototype result is superseded:
its miniature viewport was incorrect; the saved pixels have not been altered.

Implementation references: [Three.js 3D textures](https://threejs.org/docs/pages/Data3DTexture.html),
[instanced geometry](https://threejs.org/docs/pages/InstancedBufferGeometry.html),
[volume transmittance](https://pbr-book.org/4ed/Volume_Scattering/Transmittance),
[ModernGL 3D textures](https://moderngl.readthedocs.io/en/latest/reference/texture3d.html),
[framebuffer readback](https://moderngl.readthedocs.io/en/latest/reference/framebuffer.html),
and [EGL context selection](https://github.com/moderngl/glcontext).
