# SKIRTGPU implementation notes

This document tracks the incremental GPU port. The target state is a SKIRTGPU build with
one-to-one CPU parity for galaxy SEDs and IFU datacubes, including stellar, gas and dust sources,
BC03 and MAPPINGS SEDs, dust attenuation, scattering and emission.

The [2026-09-30 throughput pass](THROUGHPUT_20260930.md) targets thousands of images per
day. A production snapshot-99 galaxy at 1e8 packets now takes 4.2 s instead of about
700 s, after fixing duplicate viewing directions that disabled the resident lifecycle.
Viewer dust-emission renders go from 187 s to 50–65 s with a rewritten stochastic dust
kernel. FP16 Voronoi planes were replaced by FP32, which removed a −0.5% flux bias and a
+3–4% self-absorption bias against upstream CPU SKIRT.

The current optimization focus is optical broadband image generation. The
[2026-09-27 broadband update](OPTICAL_BROADBAND_20260927.md) adds exact spectral
rejection of unobservable detector work and persistent NVRTC compilation caching.
Two matched four-view SDSS workloads complete in 3.69 s and 2.75 s with warm
caches, 17.6x and 39.0x faster than the pristine 32-thread CPU baseline and
2.02x and 2.35x faster than the preceding GPU version. The first uncached run
still takes 7.61 s; equal image-noise parity is not established.

The [2026-09-09 z=1 JWST update](JWST_Z1_20260909.md) records the completed 1k
NIRCam/MIRI images and spectra, the validated batched stochastic-dust CUDA kernel,
bounded temporary matrix memory, and the first independent-seed packet-noise
measurements. Dust preparation is about 16x faster and the complete observation
2.7-2.8x faster than the preceding GPU version. The current 500x CPU comparison
target remains unverified; these measurements must not be multiplied by speedups
from different galaxies or mistaken for a CPU-only baseline.

The [unmodified upstream CPU test](CPU_REFERENCE_20260909.md) now measures
925 s CPU versus 181 s GPU for the same z=1 JWST model and 10 million
packets per phase: 5.11x end-to-end. Integrated band differences reach
5.60%, so whole-workflow CPU/GPU parity still needs investigation. This
result does not establish equal image noise or the 500x performance target.

## Current GPU coverage

The [BPASS/TODDLERS follow-up](BPASS_TODDLERS.md) records missing source
luminosity GPU hooks, CPU launch-worker starvation, and dust-batch sizing.
For the unchanged MOOSE-197 100M-packet observation, the selected settings
reduced GPU-version runtime from 284 to 158 s (1.80x). This is not a CPU
comparison or a GPU-only source-launch implementation.

- `skirtgpu` is built next to the regular `skirt` executable.
- `skirtgpu` enables GPU acceleration even without `-g`; CPU baselines must use
  the separate unmodified upstream `skirt` reference documented above.
- Eligible single-medium Voronoi stochastic dust source preparation now uses bounded
  GPU emissivity batches by default. `SKIRTGPU_STOCHASTIC_DUST=0` disables the path;
  `SKIRTGPU_VALIDATE_DUST_EMISSION=1` enables per-batch scalar reference checks.
- `skirt -g` enables the same runtime GPU path in the regular executable.
- `skirt -G` or `skirtgpu -G` runs a deterministic GPU backend self-test.
- The GPU backend is runtime loaded through the CUDA driver and NVRTC. CPU builds still compile and
  run if CUDA or NVRTC is missing.
- In a single-process run without `SKIRTGPU_DEVICE`, the CUDA runtime now creates one context per
  visible GPU. The resident Voronoi/table forced-scattering path round-robins expensive batch
  calls across the device pool rather than relying only on host-thread affinity. Batched Voronoi
  observer-extinction and fused HG observer-luminosity calls use the same per-call balancing because
  they feed emission and scattering peel-off. Other large batch/source/detector helpers, including
  path batches, optical-depth batches, RF reductions, SED table sampling, detector key reductions,
  albedo batches, and forced-propagation batches, use a shared balanced dispatcher as well. Scalar
  single-path helpers keep thread-local CUDA affinity. On the current two-RTX 5090 node, `skirtgpu -G`
  reports both devices and the RUN075 resident-call profile moved from a 220/20 device split to a
  116/124 split in the first 240 profiled calls. MPI runs and explicit `SKIRTGPU_DEVICE` pinning
  keep the old one-device-per-process behavior.
- The medium state is cached on the device and invalidated after setup and dynamic-state updates.
- Grid path-generation kernels exist for Cartesian, tree, adaptive mesh, tetrahedral, Voronoi,
  two- and three-dimensional cylindrical, and one-, two-, and three-dimensional spherical grids.
  These kernels are self-tested, including compact batched Voronoi traversal with nonempty
  block-list initial-cell lookup and fused dust-table optical-depth accumulation, but default
  production `-g` runs keep path generation on the CPU. The diagnostic single-path kernels can be
  enabled with `SKIRTGPU_TRACE_PATHS=1`; they are not enabled by default because per-packet CUDA
  launch and synchronization caused photon-package stalls. The opt-in batched Voronoi path is now
  faster than the earlier dense-buffer and 256-path variants on RUN075, but it is still
  host-orchestrated and far from the SKIRTGPU production target.
- Forced-scattering optical-depth accumulation for constant-section media is GPU-backed for both:
  - cumulative extinction optical depth;
  - cumulative scattering plus absorption optical depths.
  The per-segment contribution calculation, cumulative prefixing, and forced-interaction
  cumulative-path scan all run on the device.
- Non-forced interaction-point selection for constant-section media uses GPU-calculated segment
  extinction or scattering/absorption contributions, with device-side scan and CPU-identical
  interpolation of the interaction distance and absorption optical depth.
- Scattering-event albedo and normalized medium-component scattering weights for constant-section
  media are GPU-backed. These values feed packet weight adjustment, scattering peel-off weights,
  and medium-component selection before material-specific scattering.
- Dust section lookup for dust-only constant-section configurations is GPU-backed. `DustMix`
  absorption, scattering, and extinction lookup tables are flattened during medium-system setup
  and evaluated on the device for optical-depth accumulation, interaction-point selection, and
  scattering albedo/weight calculation.
- Peel-off extinction optical-depth accumulation for constant-section media is GPU-backed. This
  path includes device-side scalar reduction and thresholding, and is used by
  `FluxRecorder::detect()` before attenuated SED, image, and IFU/datacube fluxes are accumulated.
- Flux calibration for recorded SED vectors and IFU/datacube wavelength frames is GPU-backed after
  MPI reduction on the root process. Each output array is scaled by its wavelength-dependent
  calibration factor on the device, with per-array CPU fallback. The final total SED/IFU output
  arrays assembled from direct/scattered primary and secondary components are also summed on the
  device.
- Radiation-field luminosity-distance contributions for constant perceived wavelength are
  GPU-backed. The batched path can reduce segment contributions by flattened cell/wavelength RF
  bin on the device and returns only compact bin totals to the existing thread-safe host RF tables.
  This path feeds the radiation field used by dust and gas emission calculations.
- Per-cell dust absorbed luminosities for constant-section dust media are GPU-backed from the
  stored primary and stable secondary radiation-field tables. This feeds dust secondary-source
  luminosity preparation and total absorbed luminosity reporting.
- Total dust absorbed luminosity reporting for constant-section dust media uses a device-side
  reduction and returns only the primary and stable-secondary luminosity totals.
- Secondary-source launch preparation is GPU-backed for dust, continuum gas, and line gas. The
  normalized luminosity vector is scaled on the device, and the emitting-cell count and composite
  spatial-bias weight vector are computed on the device; the final history-index prefix remains on
  the CPU for exact compatibility with the current rounding behavior.
- Secondary emission uses the same contiguous batch launch entry point as primary emission. The dust
  source walks contiguous cell history-index ranges once and uses four inner launch workers per GPU
  feeder by default, preserving large device submissions while parallelizing CPU-side wavelength and
  position sampling. Supported resident runs write packed launch states while reusing one temporary
  `PhotonPacket` per worker instead of allocating one object per history.
  `SKIRTGPU_SECONDARY_LAUNCH_THREADS` controls this inner pool.
- Optical-depth accumulation has packed batch GPU kernels for extinction-only and
  scattering/absorption paths. Constant-section media and dust-section lookup tables are both
  covered; the table path accepts one wavelength per path. These kernels accept variable segment
  counts per path and write the cumulative optical depths back to the existing `SpatialGridPath`
  segment layout.
- `MediumSystem` exposes batch helpers that prepare complete photon-packet paths and invoke the
  packed dust-table optical-depth kernels. These are the production-facing bridge for replacing
  the scalar forced-scattering path preparation inside the Monte Carlo packet loop.
- Voronoi path preparation can batch complete photon-packet paths on the GPU before the
  packed dust-table optical-depth kernels run. This is currently limited to Voronoi grids. The batch
  path uses a compact two-pass layout: one GPU launch counts actual segments per photon, and a
  second launch writes only the compact cell/length arrays needed to reconstruct the paths. The
  initial-cell lookup uses the CPU Voronoi search-grid block lists on the device and falls back to a
  full site scan only when block data is unavailable.
- When the first forced-scattering batch uses Voronoi grids and GPU dust-section lookup
  tables, path generation and cumulative dust-table optical-depth accumulation are fused into the
  compact output pass. This avoids sending reconstructed `SpatialGridPath` segments back to a
  second GPU optical-depth kernel, while keeping the existing CPU path object as the compatibility
  boundary for radiation-field storage, forced-propagation, and later scalar scattering events.
- For Voronoi grids with dust-section lookup tables, the production forced-emission path can keep
  the complete store-radiation-field plus forced-propagation step resident on the GPU. The default
  route traces each photon path once, reduces radiation-field luminosity-distance sums with a
  device hash, samples the forced interaction optical depth on the GPU from CPU-generated uniform
  deviates, evaluates table albedos on the device, and returns only compact RF sums plus one
  forced-propagation result per packet. Resident buffers are reused as per-device high-water scratch
  storage, and path offsets are now built on the device from counted Voronoi path lengths so the
  hot path no longer copies the full count vector back to the host just to build offsets. This is
  used before falling back to the exact resident path or the prepared-path batch kernels.
- The resident Voronoi/table path now accumulates compact radiation-field sums directly
  into a persistent per-GPU dense RF table by default. This avoids
  copying millions of compact RF keys/sums back to the CPU and avoids the host lock-free RF table
  update in the hot photon-package path. The GPU accumulators are flushed into the host RF tables
  at `communicateRadiationField()` before the existing process-level sum. Set
  `SKIRTGPU_RF_ACCUMULATOR_DIRECT=0` to force the older host RF-store path.
- Radiation-field luminosity-distance contributions have packed batch GPU kernels for cumulative
  optical-depth paths with constant perceived wavelength. The current fast path computes
  per-segment contributions, inserts them into a device hash keyed by flattened RF bin, compacts
  occupied bins on the device, and then applies those compact sums to the existing lock-free host
  radiation-field tables.
- Forced-propagation interaction-point lookup in cumulative optical-depth paths has both scalar and
  packed batch GPU kernels. The packed form handles variable segment counts per path and is
  self-tested against CPU interpolation. Constant-section and dust-section table scattering
  albedos also have batch kernels, so the batched forced-scattering path can avoid one scalar
  material lookup per packet after the interaction points are known. Higher-level packed
  forced-propagation kernels compute the interaction point and luminosity bias factor in one launch
  for explicit absorption and for non-explicit dust-table paths without moving media. These kernels
  are wired into the first forced-scattering batch used by production `-g` runs when the medium
  supports it.
- Forced-scattering rounds can be processed in packet waves after the first propagation step.
  Every subsequent path/tau preparation, radiation-field contribution, and forced propagation step
  is batched again. For supported single-dust, nonmoving, unpolarized configurations, lifecycle
  waves of at least 65536 packets now also use the direct HG scattering peel-off batch by default.
  The fused CUDA operation evaluates the phase function and observer extinction together. A custom
  two-dimensional kernel processes every exact unique observer direction in one launch and uploads
  the packet state once per scattering round. The production path also projects matching Frames,
  bins disjoint-grid SEDs, applies optional SED apertures, and writes directly to per-instrument
  persistent GPU accumulators in that launch. Observer luminosities, detector keys, and detector
  values no longer make a GPU-to-CPU-to-GPU round trip. Smaller parent lifecycle waves retain the
  scalar scattering peel-off because the per-round GPU launch and transfer overhead is larger than
  the saved CPU work.
- Supported single-dust, nonmoving, unpolarized Voronoi workloads now run complete packet lifecycles
  through the resident mixed-precision CUDA path by default for waves of at least 65536 packets.
  Initial emission detection, analog propagation, later HG scattering, Frame/SED accumulation, and
  termination remain on the selected GPU. Final secondary-emission passes can use the same path
  without allocating or updating a radiation-field accumulator; this removes the old prepared-path
  and host-driven peel-off fallback from the secondary shooting phase.
- The mixed-precision lifecycle caches normalized FP32 bisector planes for every Voronoi neighbor
  edge, avoiding repeated site-coordinate subtraction and plane reconstruction in the innermost
  packet and detector traversal. Imported-source waves also find each packet's initial Voronoi cell
  while preparing wavelength lookup pairs, avoiding a second full initial-cell search during
  emission peel-off. When `minScattEvents` is zero, the lifecycle omits the scattering-count buffer
  and its initialization and update traffic. After each propagation round, the compacted survivor
  list is fed back into the random-generation and transport kernels so terminated packet slots are
  no longer relaunched.
- Scattering peel-off profiling is split from material scattering in the forced-round controller.
  `SKIRTGPU_PROFILE_BATCH=1` separates initial emission peel-off from later scattering peel-off and
  reports batch versus scalar time. In a 70 second, 268435456-packet two-RTX-5090 profile on
  `TNG50-1_99_746970_sdss_broadband.ski`, direct fused HG peel-off plus the multi-observer kernel
  first reduced mean total time from 43.6 seconds to 14.4 seconds per 262144-packet lifecycle wave.
  Fusing Frame and SED accumulation into the same multi-observer launch, then reusing the validated
  immutable grid topology, reduced the complete direct observer/detector group from 1481 ms to
  428 ms with the same uncapped scheduler. The automatic eight-feeder scheduler reached 12.4% in a
  matched 60-second window versus 9.4% before detector fusion. Resident propagation contention now
  limits the end-to-end gain. The GPU accumulators, key reductions, and Voronoi traversal reuse
  persistent per-runtime high-water storage, and exactly matching noncontiguous Frame/SED observer
  directions are traced only once.
- Total-only `FrameInstrument` outputs with `BandWavelengthGrid` have an opt-in frame-band detector
  kernel (`SKIRTGPU_FRAME_BAND_DETECTOR=1`) that projects packet positions, interpolates band
  transmission curves, applies extinction, and reduces frame pixels by key on the GPU. It is
  self-tested and smoke-tested at roundoff parity, but it is not enabled by default because the
  20 June 2026 RUN075 profile was neutral to slightly slower than the scalar detector path.
  A direct accumulator variant (`SKIRTGPU_FRAME_BAND_DETECTOR_DIRECT=1`) keeps those reduced
  detector additions on the GPU until flush. The HG observer-luminosity and frame-band detector
  wrappers now reuse high-water CUDA scratch buffers instead of allocating/freeing temporary buffers
  on every diagnostic scatter-peel call, but the direct path still remains a diagnostic switch only.
- Initial emission peel-off for distant instruments can batch the observer-direction extinction
  optical-depth calculation on the GPU in the first forced-scattering batch. For Voronoi
  grids with dust-section tables, the observer path now uses a totals-only fused Voronoi/table
  kernel that returns one optical depth per peel-off packet without materializing path segments on
  the host. Direct frame-instrument emission peel-off passes the single distant-observer direction
  through a shared-direction overload, avoiding per-batch host `SpatialGridPath` construction and
  avoiding a temporary direction vector. Total-only `SEDInstrument` outputs now participate in the
  same direct batch dispatch as `FrameInstrument` outputs. A custom opt-in frame accumulator kernel
  can return the optical depth it already calculated so a matching SED consumes the same observer
  trace rather than launching another one. On the large 262144-packet profile this reduced direct
  emission peel-off from 1.724 seconds to 1.403 seconds per wave, but it remains opt-in because the
  5-million-packet RUN000 case regressed from 37.0 seconds to 39.3 seconds overall.
- Imported BC03 and MAPPINGS source luminosity setup can evaluate stored-table CDF normalization
  factors in a single GPU batch over all imported entities. The CPU scalar `SEDFamily::cdf()` path
  remains the fallback for unsupported SED families or unavailable GPU runtime.
- High-count imported BC03 and MAPPINGS primary sources now use a custom CUDA source-launch kernel.
  It maps history indices to source entities and samples the entity SED, optional logarithmic
  wavelength bias, cubic-spline particle position, and isotropic direction on both GPUs. A reusable
  per-entity FP32 PDF/CDF cache is uploaded once per GPU, while wavelengths, luminosities, and the
  packed launch result remain FP64.
- Supported imported primary sources now initialize the resident lifecycle directly on the selected
  GPU. Source entity, wavelength, position, direction, luminosity, radiation-field bin, and random
  state are generated into the resident structure-of-arrays buffers, so primary packets do not make
  a source-device-to-host-to-lifecycle-device round trip. Unsupported source configurations retain
  the packed launch bridge, which avoids transient `PhotonPacket` objects. The older per-packet GPU
  spectrum-precomputation path remains diagnostic only.
- Supported single-dust, nonmoving, unpolarized Voronoi simulations also initialize dust-secondary
  packets directly in the resident lifecycle buffers. A reusable per-cell FP32 spectrum cache and
  history-to-cell map are uploaded once per GPU; the CUDA source kernel samples wavelength,
  luminosity, isotropic direction, and a position inside the emitting Voronoi cell. Position sampling
  rejection-tests the cell bounding box against the resident Voronoi neighbor topology. Unsupported
  dust configurations and gas secondary sources retain the packed launch bridge.
- `SpatialGridPath` segment storage is now reserved lazily on first `addSegment()` instead of in
  the constructor. This avoids allocating a 1000-segment path buffer for every `PhotonPacket` in a
  GPU resident batch before any host path is materialized, while preserving the reuse benefit for
  scalar packets once they actually store path segments.

These paths are used by the photon lifecycle before SED and datacube instruments receive peeled-off
packets, so they are part of the production ray-tracing flow rather than a standalone prototype.

## Runtime controls

- `SKIRTGPU=1` or `SKIRT_GPU=1`: enable GPU acceleration for the process.
- `SKIRTGPU_DEVICE=<index>`: select one CUDA device explicitly. This disables the single-process
  all-device pool and is useful for per-GPU debugging.
- `SKIRTGPU_USE_ALL_DEVICES=0`: opt out of the single-process all-device pool. By default,
  single-process GPU runs without `SKIRTGPU_DEVICE` use all visible CUDA devices. MPI runs keep
  one selected device per process.
- `SKIRTGPU_MAX_DEVICES=<count>`: cap the number of visible CUDA devices used by the single-process
  all-device pool.
- `SKIRTGPU_MIN_SEGMENTS=<count>`: minimum path length for GPU optical-depth and radiation-field
  offload. The default is 64 segments.
- `SKIRTGPU_TRACE_PATHS=1`: opt into the diagnostic single-photon grid path kernels. This is off
  by default because the current path kernels launch and synchronize one CUDA thread per photon,
  which is much slower than CPU traversal for production photon-package runs until batched
  transport is implemented.
- `SKIRTGPU_SYNC_PHOTON_CYCLE=1`: opt into the current synchronous per-photon optical-depth,
  interaction-point, scattering-weight, and radiation-field kernels. This is off by default for
  production `-g` runs because these kernels launch and synchronize once per packet/path; the bulk
  GPU reductions, launch-preparation kernels, and flux-output kernels remain enabled.
- `SKIRTGPU_BATCH_FIRST_FORCED=0`: opt out of the chunk-level forced-scattering path used by
  production `-g` runs when the simulation has a medium with forced scattering. When enabled, each
  Monte Carlo progress chunk launches packets, batches initial emission peel-off extinction for
  distant instruments, then tries the resident Voronoi/table radiation-field plus forced-
  propagation GPU pass before falling back to prepared-path batch kernels.
- `SKIRTGPU_RESIDENT_LIFECYCLE=0`: opt out of the complete resident packet lifecycle. It is enabled
  by default for supported waves of at least `SKIRTGPU_RESIDENT_MIN_PACKETS` packets (65536 by
  default), including final secondary-emission passes that do not store another radiation field.
- `SKIRTGPU_RESIDENT_MIXED_PRECISION=0`: use the slower double-precision resident geometry path.
  Normalized Voronoi geometry and density traversal use FP32 by default while packet luminosities,
  wavelengths, detector accumulators, and radiation-field accumulators remain FP64.
- `SKIRTGPU_RESIDENT_ANALOG_TRANSPORT=0`: use forced interaction sampling in the resident lifecycle.
  The default analog transport samples one untruncated optical depth per scattering event and avoids
  a second traversal of each path.
- `SKIRTGPU_RESIDENT_SAMPLE_OBSERVER=0|1`: control one-observer sampling for initial-emission and
  scattering peel-off. It is enabled automatically for resident waves containing at least 1,048,576
  packets and disabled below that threshold. The sampled contribution is multiplied by the observer
  count, preserving the expectation value while increasing Monte Carlo variance. Explicit `0`
  disables sampling and explicit `1` forces it for every supported wave.
- `SKIRTGPU_RESIDENT_ASYNC_ROUNDS=1`: enqueue fixed groups of resident lifecycle rounds before a
  completion check. This remains opt-in because the tested groups were slower than synchronized
  active-wave compaction.
- `SKIRTGPU_RESIDENT_ASYNC_ROUND_GROUP=<count>`: set the number of rounds in each asynchronous group;
  the default is 32. Groups of 2 and 32 both regressed the high-count benchmark, so asynchronous
  rounds remain disabled by default.
- `SKIRTGPU_LIFECYCLE_BLOCK_SIZE=<count>`: override the resident transport/scatter CUDA block size.
  Values must be multiples of 32 from 32 through 1024; the tuned default is 320.
- `SKIRTGPU_LIFECYCLE_DETECTOR_BLOCK_SIZE=<count>`: independently override the resident detector
  block size with the same validation. The tuned default is 256.
- `SKIRTGPU_SOURCE_LAUNCH=0`: disable the default custom CUDA imported-source launch kernel and use
  scalar CPU packet generation.
- `SKIRTGPU_SOURCE_LAUNCH_MIN_PACKETS=<count>`: set the minimum packet count per imported source
  component for the custom CUDA source-launch kernel. The default is 4,194,304, retaining the CPU
  path where small workloads cannot amortize source-cache construction and transfer.
- `SKIRTGPU_PACKED_SOURCE_LIFECYCLE=0`: disable the default packed primary/dust-secondary launch
  bridge and construct a full `PhotonPacket` object for every history before resident submission.
  The fallback remains automatic for moving or polarized dust, gas sources, unsupported primary
  sources, packet-launch callbacks, and nonresident detector or medium configurations.
- `SKIRTGPU_DEVICE_SOURCE_LIFECYCLE=0`: disable direct same-device imported-source initialization
  and use the packed source-to-lifecycle bridge. The fallback is automatic when a lifecycle wave
  crosses source components or uses an unsupported source or callback.
- `SKIRTGPU_DEVICE_DUST_SOURCE_LIFECYCLE=0`: disable direct same-device dust-secondary initialization
  and use the packed source-to-lifecycle bridge. Direct dust launch currently requires a single dust
  medium on a Voronoi grid, no moving media or spheroidal polarization, one cell per library entry,
  and either no wavelength bias or logarithmic wavelength bias.
- `SKIRTGPU_PRECOMPUTE_SOURCE_SPECTRA=1`: opt into imported-source GPU spectrum precomputation.
  It is disabled by default because the 6.8-million-packet benchmark spent about 46 seconds in this
  step after logging `Launching ... photon packets`, which looked like a transport stall and was much
  slower than on-demand source spectrum caching.
- `SKIRTGPU_SECONDARY_LAUNCH_THREADS=<count>`: set the inner dust-secondary packet-construction pool
  per GPU feeder. The GPU default is four (maximum eight); CPU-only runs retain the regular scheduler.
- `SKIRTGPU_RESIDENT_FORCED=0`: opt out of the resident Voronoi/table store-radiation-field plus
  forced-propagation pass and use the older prepared-path batch path.
- `SKIRTGPU_RESIDENT_SAMPLE_ON_GPU=0`: keep the resident Voronoi/table pass but sample the forced
  interaction optical depth after a separate GPU total-optical-depth pass. This exact resident
  fallback is useful for debugging; on the current RUN075 profile it was slower than the default
  sampled resident path.
- `SKIRTGPU_BATCH_FIRST_FORCED_CHUNK=<count>`: photon-packet chunk size used by the
  forced-scattering batch. The default is 8,388,608 for production `-g` runs. A 4 September 2026
  sweep on the 536.9-million-packet two-RTX-5090 benchmark found 4,194,304 and 8,388,608 packets
  faster than 1,048,576 or 2,097,152, with no measurable primary gain at the larger setting.
  Setting this to 1000 is useful for
  matched-order regression checks against older smoke baselines; setting it back to 262144 is useful
  for A/B comparisons with the preceding production default.
- `SKIRTGPU_LIFECYCLE_THREADS=<count>`: override the CPU feeder-thread limit for all GPU-enabled
  photon-lifecycle work; `0` keeps the regular uncapped SKIRT scheduler.
- `SKIRTGPU_PRIMARY_LIFECYCLE_THREADS=<count>` and
  `SKIRTGPU_SECONDARY_LIFECYCLE_THREADS=<count>`: independently override the primary and secondary
  feeder limits, taking precedence over `SKIRTGPU_LIFECYCLE_THREADS`. Without an override, segments
  containing at least 4,194,304 packets use six feeders for direct-device primary emission, four for
  direct-device dust-secondary emission, and twelve for CPU-generated packed secondary emission.
  Smaller segments remain uncapped. The threshold leaves low-packet workloads on the regular
  scheduler, where CPU execution can be faster after GPU setup and launch overhead are included.
- `SKIRTGPU_LIMIT_LIFECYCLE_THREADS=0`: ignore `SKIRTGPU_LIFECYCLE_THREADS` even if it is set.
- `SKIRTGPU_BATCH_FORCED_ROUNDS=0`: opt out of batched later forced-scattering rounds after the
  first propagation step. This is enabled by default for production `-g` runs because it improves
  RUN075 throughput, but the larger packet waves change random-stream interleaving across packets.
- `SKIRTGPU_COMPACT_ROUND_QUEUE=1`: opt into a compact host-side survivor queue for later
  forced-scattering rounds. The default keeps the previously benchmarked full-chunk scan. On the
  post-RF-accumulator RUN075 profile it slightly improved some 65536-packet chunk timings but did
  not improve the capped progress, and combining it with the 262144-packet wave was slightly
  slower than the promoted default, so it remains opt-in.
- `SKIRTGPU_BATCH_SCATTER_PEEL_DIRECT=0|1`: control the direct HG scattering peel-off batch for
  supported distant total-only instruments. With no override it is enabled for every scattering
  round whose parent lifecycle wave contains at least 65536 packets, and disabled for smaller
  parent waves. Explicit `1` forces it for diagnostics and explicit `0` disables it.
- `SKIRTGPU_BATCH_SCATTER_PEEL_FUSED=0`: disable the fused Voronoi/table HG
  observer-luminosity kernel used by the direct path. It is enabled by default whenever direct HG
  peel-off is selected, computing the phase factor and observer extinction in one GPU pass.
- `SKIRTGPU_BATCH_SCATTER_PEEL_MULTI_OBSERVER=0`: disable the custom multi-observer fused HG kernel
  and fall back to one fused launch per exact unique observer. The multi-observer route is enabled
  by default whenever fused direct HG peel-off is selected.
- `SKIRTGPU_BATCH_SCATTER_PEEL_DETECTOR_FUSED=0`: disable direct Frame/SED accumulation inside the
  multi-observer HG kernel and return observer luminosities to the host detector wrappers. This is
  enabled by default when the direct scattering path is selected. It currently supports total-only
  broadband Frames and total-only SEDs using disjoint wavelength grids, including circular SED
  apertures.
- `SKIRTGPU_BATCH_SCATTER_PEEL=1`: opt into the older generic host-side scattering peel-off batch
  when the direct HG path is disabled or unsupported. This remains diagnostic.
- `SKIRTGPU_BATCH_HG_SCATTER=1`: opt into the standalone GPU Henyey-Greenstein random-walk
  scattering direction sampler for the single dust medium, no moving media, unpolarized case. This
  kernel passes the GPU self-test but is off by default because the host-launched per-round wrapper
  is slower than the CPU loop: on 20 June 2026, a 45 second RUN075 window showed median
  `scatter_ms` around 801 ms with this option versus 11 ms with the default CPU scatter step. The
  kernel is retained as a primitive for a future resident GPU packet queue where it can be fused
  without per-round host copies.
- `SKIRTGPU_RESIDENT_HG_SCATTER=1`: opt into the resident fused variant of the same HG
  random-walk direction sampler. The sampled resident forced-propagation pass returns the next
  outgoing direction, and the host applies it after scattering peel-off in the following round.
  This preserves the physical peel-off/scatter order and avoids the standalone scatter kernel's
  per-round setup cost. It is off by default because the 20 June 2026 RUN075 comparison was
  throughput-neutral: median `scatter_ms` dropped from 11.65 ms to 3.81 ms, but 65 second progress
  was 4.9% versus 5.1% for the default build.
- `SKIRTGPU_BATCH_VORONOI_PATHS=0`: opt out of batched GPU Voronoi path generation for the batch
  optical-depth helpers. This keeps CPU fallback if the grid is not Voronoi or if the GPU path
  fails. On the current RUN075 test this compact block-list path advances correctly and improves
  the earlier dense-buffer and 256-path GPU variants, but remains limited by CPU reconstruction of
  `SpatialGridPath` objects and host-driven photon lifecycle orchestration.
- `SKIRTGPU_BATCH_VORONOI_PATHS_MAX=<count>`: maximum number of photon paths per Voronoi path
  generation launch. The default is 1024, which keeps the compact count/write batches bounded while
  avoiding the earlier worst-case per-photon segment buffers.
- `SKIRTGPU_FRAME_BAND_DETECTOR=1`: opt into the total-only frame-band detector projection and
  band-transmission GPU kernel. The kernel passes self-tests and smoke parity checks, but it is off
  by default because it did not improve RUN075 throughput in the current host-orchestrated pipeline.
- `SKIRTGPU_FRAME_BAND_DETECTOR_DIRECT=1`: opt into the direct frame-band detector accumulator.
  This exercises the GPU-resident detector add path, but it is diagnostic-only because it increased
  initial peel-off cost on RUN075.
- `SKIRTGPU_DETECTOR_ACCUMULATOR=1`: opt into a generic detector accumulator after host-side
  detector key/value preparation. This avoids host `LockFree::add()` for total-only distant Frame
  and SED outputs, but it is still diagnostic while the photon-package loop remains host
  orchestrated.
- `SKIRTGPU_PROFILE_BATCH=1`: print per-chunk timing for the batched forced-scattering path. The
  stage timings are intended for development profiling and are off by default. When the combined
  radiation-field/forced-propagation pass is used, its time is reported as `store_forced_ms`
  instead of separate `store_ms` and `forced_ms` entries. The peel-off fields are split into
  `emission_peel_batch_ms`, `emission_peel_scalar_ms`, `later_peel_batch_ms`, and
  `later_peel_scalar_ms` so the initial emission peel-off and later scattering peel-off paths can be
  profiled independently.
  `SKIRTGPU_PROFILE_BATCH_LIMIT=<count>` caps the number of batch-profile log entries; `0` means
  uncapped.
- `SKIRTGPU_PROFILE_RESIDENT=1`: print opt-in low-level timings from the resident Voronoi/table
  radiation-field plus forced-propagation CUDA path. `SKIRTGPU_PROFILE_RESIDENT_LIMIT=<count>` caps
  the number of resident calls printed; the default limit is 64. This is a diagnostic stderr stream,
  not normal SKIRT logging.
- `SKIRTGPU_PROFILE_STORE_FORCED=1`: print opt-in host-side timings around the combined
  store-radiation-field plus forced-propagation path, including random draw, resident GPU wrapper,
  RF table store, packet apply, and total time. `SKIRTGPU_PROFILE_STORE_FORCED_LIMIT=<count>` caps
  the number of log entries; `0` means uncapped.
- `SKIRTGPU_PROFILE_SCATTER_PEEL=1`: print opt-in substage timings for the direct/fused HG
  scattering peel-off diagnostic path, split into host preparation, observer luminosity/extinction,
  detector accumulation, and total time. `SKIRTGPU_PROFILE_SCATTER_PEEL_LIMIT=<count>` caps the
  number of log entries; `0` means uncapped.
- `SKIRTGPU_SERIAL_RF_STORE=1`: opt into a serialized bulk radiation-field update for GPU compact
  RF reductions. This protects each compact GPU result with one mutex and performs plain additions
  instead of per-bin CAS loops. It is off by default because the RUN075 two-GPU profile showed that
  serializing competing chunk updates worsened aggregate `store_forced_ms`.
- `SKIRTGPU_RF_ACCUMULATOR=1`: opt into a diagnostic host-to-device RF accumulator. It receives
  compact RF sums after they have already been copied back to the host, sends those compact arrays
  back to a dense per-GPU accumulator, and flushes at `communicateRadiationField()`. It is retained
  for A/B testing but is not promoted because the extra round-trip did not improve RUN075 chunk
  throughput.
- `SKIRTGPU_RF_ACCUMULATOR_DIRECT=0`: opt out of direct resident RF accumulation. In supported
  resident Voronoi/table RF-plus-forced batches, compact RF sums are now accumulated into the
  per-GPU dense RF table before any host compact-array copy. The normal host RF store receives an
  empty compact vector for that batch, and the accumulators are flushed into the ordinary RF tables
  at the existing communication boundary.
- `SKIRTGPU_PROFILE_LAUNCH=1`: print opt-in imported-source batch launch timings, split into
  wavelength selection, position generation, launch-direction conversion, `PhotonPacket`
  initialization, and overhead. `SKIRTGPU_PROFILE_LAUNCH_LIMIT=<count>` caps the number of stderr
  entries; the default limit is 8.
- `SKIRTGPU_RESIDENT_RF_ONLY=0`: keep radiation-field-only passes (dust self-absorption
  iterations without peel-off) on the host path. By default they run resident with a
  detector-less observer when packets are initialized on the GPU.
- `SKIRTGPU_SHARE_OBSERVER_TRACES=0`: trace each observer slot separately even when several
  slots share a direction (default: one shared extinction ray per direction).
- `SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS=<count>`: cells per stochastic dust GPU launch,
  1–1048576; by default about 1/16 of the emitting cells per pass, clamped to 2048–16384. The
  warp-per-cell kernel needs no per-cell scratch memory.
- `SKIRTGPU_DUMP_DUST_BATCH=<prefix>`: write four stochastic dust kernel batches for the
  standalone benchmark harness.
- `SKIRTGPU_NVRTC_LIBRARY=<path>`: override the NVRTC shared library path.
- `SKIRTGPU_NVRTC_BUILTINS_LIBRARY=<path>`: override the NVRTC builtins shared library path.

## Multi-GPU use

For a single-process `skirtgpu -g` or `skirt -g` run, the runtime now uses all visible CUDA devices
unless `SKIRTGPU_DEVICE` is set or `SKIRTGPU_USE_ALL_DEVICES=0` is exported. Worker threads are
assigned to CUDA contexts across the pool, and the resident forced-scattering path uses its own
round-robin dispatcher so independent batch calls can run on both GPUs even when host scheduling is
skewed. Batched observer-extinction and fused HG observer-luminosity kernels are also balanced
per call, and large batch/source/detector helper kernels use a shared per-call dispatcher. This
keeps the existing CPU-side Monte Carlo scheduling and is therefore a throughput improvement, not a
full device-resident photon pipeline.

MPI remains supported. With multiple MPI ranks, each rank selects one CUDA device as
`rank % device_count`, so a two-rank run on a two-GPU node uses both GPUs without the single-process
pool:

```sh
mpiexec -n 2 skirtgpu -t 1 -k -o <output-dir> <simulation.ski>
```

The `-s` option runs multiple simulations inside one process; it can use the single-process pool,
but it does not create MPI ranks. Use `SKIRTGPU_DEVICE=<index>` only when manually pinning a
process to a specific device, and `SKIRTGPU_MAX_DEVICES=<count>` when reserving part of a node.

## Performance target

The SKIRTGPU target for production galaxy SED/IFU workloads is at least 1000x speedup relative to
the CPU SKIRT version. The supported production scope now has resident batched propagation, grid,
material, detector state, imported primary-source sampling, and dust-secondary generation. Meeting
the target still requires lower-overhead survivor scheduling and detector accumulation, broader
resident material/grid coverage, and removal of runtime CUDA compilation from measured startup.

The current path is stable and substantially faster, but it is not yet a verified 1000x result. On
4 September 2026, the promoted two-RTX-5090 defaults completed the matched 6,822,200-primary plus
6,822,200-secondary `TNG50-1_99_706717_sdss_broadband.ski` photon run in 2.056 seconds (1.747 seconds
primary and 0.309 seconds secondary), versus 44.622 seconds for 32-thread CPU SKIRT. This is a 21.7x
photon-shooting speedup. Full simulation time was 6.5 seconds versus 46.4 seconds, or 7.1x, because
CUDA/NVRTC setup still takes about four seconds. Against the CPU reference, integrated SED totals
differed by 0.069-0.095%, integrated FITS totals by 0.027-0.136%, and the worst individual FITS-band
total by 0.607%; every checked value was finite.

The high-packet-count `TNG50-1_99_746970_sdss_broadband.ski` case completed 268,435,456 primary and
268,435,456 secondary packets with direct same-device primary and dust-secondary initialization,
direct-device waves, phase-specific feeder limits, sampled high-count observers, precomputed
Voronoi planes, survivor-fed transport, and tuned transport/detector blocks. Six unprofiled runs
measured 2.985-3.355 seconds for primary emission, 1.335-1.349 seconds for secondary emission, and
4.322-4.704 seconds for combined photon shooting. Mean shooting time was 4.430 seconds, or 121.2
million launched packets per second across both GPUs. Mean end-to-end time was 8.7 seconds. The
preceding packed source-to-lifecycle path required 24.0 seconds (12.7 plus 11.3 seconds), so the
current path is 5.42x faster than that GPU implementation. The complete matched 32-thread CPU run
required 1458.065 seconds of photon shooting (1270.711 seconds primary plus 187.354 seconds
secondary), giving mean primary, secondary, and combined shooting speedups of 411.4x, 139.6x, and
329.1x. Reported end-to-end time was 1459 seconds on CPU versus a mean 8.7 seconds on both GPUs, or
about 167.7x. Against that CPU reference, integrated SED totals differed by 0.0022-0.0376%, individual
integrated FITS bands differed by at most 0.230%, all checked values were finite, and the convergence
reports were byte-identical. Reaching 1000x still requires further resident transport/detector
acceleration and lower startup overhead.

The Voronoi-plane cache reduced profiled transport time by 6.2% for primary packets and 15.5% for
secondary packets. Feeding compacted survivors back into transport then reduced high-count shooting
from 5.639 to 5.123 seconds. The next pass reused each dust packet's known launch cell in emission
peel-off, added a dedicated emission detector, stopped allocating and uploading an all-infinity
observer-distance array, and exactly rejected final-secondary rays whose sampled optical depth
exceeded a conservative global or direction-to-box-exit column-density upper bound. The bounded
no-radiation-field kernel is separate from primary transport so its extra state does not reduce
primary occupancy. Together these changes reduced mean high-count shooting from 5.123 to 4.430
seconds. Relative to the preceding promoted GPU output, SED values changed by less than
3e-8 percent, peak-normalized FITS differences were below 5.1e-6 percent, all FITS values remained
finite, and convergence reports were byte-identical. A custom cell-histogram, prefix-scan, and
scatter queue was also tested:
cell ordering helped fixed-observer traversal but hurt random-direction transport, and an alternating
repeat measured 5.202 seconds with detector-only cell ordering versus 5.120 seconds without it. The
cell-binned queue therefore remains unpromoted.

A follow-up extended the Voronoi-plane cache to detector rays, reused the initial-cell lookup for
direct imported-primary packets, and omitted scattering-count storage when `minScattEvents` is zero.
Three additional unprofiled 536.9-million-packet runs measured 3.038-3.159 seconds for primary,
1.342-1.354 seconds for secondary, and 4.386-4.501 seconds for combined shooting, with a 4.434-second
mean. This confirms the 329x combined shooting result within run variance. Integrated SED totals
again differed from CPU by 0.0022-0.0376%, individual integrated FITS bands by at most 0.230%, all
checked values were finite, and the convergence report was byte-identical. An observer-major cache
of reciprocal detector plane-direction dot products was rejected: it increased profiled primary
detector time from 997 ms to 1178 ms and primary emission-accounted work from 849 ms to 949 ms because
the extra global-memory traffic outweighed the saved divides.

For historical comparison, on 20 June 2026, a single-process two-GPU bounded run on
`RUNs_snap_075_cfhtmegacam_sdss_z005.ski` advanced 6.9% of 114,032,000 primary photon packets
before a 65 second timeout, after about 6.8 seconds of setup. The default chunk for this run was
65536 packets. A 32768-packet two-GPU run advanced 5.1%, and the same sampled-resident path pinned
to `SKIRTGPU_DEVICE=0` previously advanced 2.7% before the timeout. Median profiled timings changed
as follows:

- single GPU: `store_forced_ms` 4185 ms, `peel_ms` 1166 ms, `total_ms` 6017 ms;
- two GPUs before resident scratch/offset reuse, 32768 packet chunks: `store_forced_ms` 1705 ms, `peel_ms` 518 ms,
  `total_ms` 2904 ms;
- previous two-GPU 32768-packet default: `store_forced_ms` 1634 ms, `peel_ms` 492 ms,
  `total_ms` 2834 ms;
- current two-GPU 65536-packet default: `store_forced_ms` 1992 ms, `later_peel_ms` 1186 ms,
  `scatter_ms` 29 ms, `total_ms` 4403 ms per larger packet chunk.

On the same current build, `SKIRTGPU_RESIDENT_HG_SCATTER=1` reached 4.9% in a 65 second RUN075
window, with median `scatter_ms` 3.81 ms, `store_forced_ms` 1648 ms, and `total_ms` 2868 ms. The
default reached 5.1%, with median `scatter_ms` 11.65 ms, `store_forced_ms` 1691 ms, and
`total_ms` 2878 ms. Because progress did not improve, the fused scatter direction output remains
opt-in until it can be folded into a larger resident packet-state loop. The opt-in frame-band
detector kernel reached only 2.6% in the same 65 second RUN075 window. Opt-in lifecycle feeder
caps improved individual chunk latency but did not beat the uncapped default throughput. The
batched-round modes are throughput scaffolding and do not yet preserve CPU-identical random-stream
interleaving.

After the direct observed-extinction overloads were added on 20 June 2026, the same RUN075 SDSS
smoke path reached primary emission, printed capped batch profiles, and advanced to 4.1% before a
55 second timeout. The first capped chunks showed `peel_ms` around 560-955 ms, with warm source
launch still dominated by GPU queueing rather than scalar packet creation. A detailed
`SKIRTGPU_PROFILE_STORE_FORCED` sample showed the resident forced path was active (`combined=yes`)
and individual resident store/forced calls took about 0.13-0.40 seconds, so the remaining
multi-second chunk timings are mostly host-thread queueing onto the two GPU runtimes plus later
scatter peel-off work, not a GPU deadlock.

The primary-source batch launch API was added in the same iteration. It compiled and passed
`skirtgpu -G`, and the RUN075 default capped-profile smoke still advanced to 4.0% before the
55 second timeout. The first profiled chunks continued to show cold/queued `launch_ms` near
2.9 seconds, so the immediate speed limit is not the source/entity upper-bound lookup itself.

The next 20 June 2026 packet-shooting pass found that most of that `launch_ms` was not source
lookup or wavelength sampling. It came from constructing a 65536-element `vector<PhotonPacket>`:
each packet inherited a `SpatialGridPath` constructor that eagerly reserved 1000 path segments.
After making that segment reserve lazy and adding a small GPU isotropic-direction primitive,
`skirtgpu -G` passed and the RUN075 SDSS capped profile dropped the first eight `launch_ms` values
from about 2.99-3.12 seconds to about 0.18-0.32 seconds per 65536-packet chunk. The corresponding
`SKIRTGPU_PROFILE_LAUNCH` lines showed the imported-source launch work itself at about 0.09-0.21
seconds per chunk. The same clean 55 second smoke run reached 3.9% primary-emission progress; the
dominant timings are now again `store_forced_ms` and later scattering peel-off, not packet object
construction.

The follow-up profile showed individual resident CUDA calls were tens of milliseconds, while the
host RF table update from compact GPU sums took roughly 90-170 ms per call because it performed
millions of lock-free compare-and-swap additions. A serialized plain-add bulk RF store was tested
as `SKIRTGPU_SERIAL_RF_STORE=1`: it reduced some individual `rf_store_ms` samples but caused chunk
queueing and worsened aggregate `store_forced_ms`, so the lock-free default remains promoted.

The next RF-store pass added a persistent per-GPU RF accumulator. The first host-to-device variant
(`SKIRTGPU_RF_ACCUMULATOR=1`) was correct but not faster because it still copied compact RF sums
from device to host and then sent them back to the GPU; a 45 second RUN075 profile showed first
chunk `store_forced_ms` values around 2.4-4.1 seconds, comparable to or worse than the default.
The direct resident variant now accumulates `dCompactKeyv` and `dCompactSumv` on device before the
compact arrays are copied out. On the same RUN075 SDSS profile,
the first eight store-forced calls reported `bins=0`, `rf_store_ms=0`, and total inner
store-forced times of about 52-166 ms. The first profiled packet chunks had `store_forced_ms` of
about 0.75 seconds and 1.07 seconds before later queueing effects, versus multi-second default
chunks. The capped direct run reached 2.1% primary progress in 45 seconds; the comparable
non-direct accumulator run reached only 0.1% before the same cap. A full
`RUNs_snap_000_sdss_broadband.ski` validation with direct accumulation completed primary,
secondary, and final output in 43.9 seconds, exercising both RF communication flushes without an
accumulator error. This is now the default path for supported `-g` emission batches; set
`SKIRTGPU_RF_ACCUMULATOR_DIRECT=0` only for A/B comparisons. After promotion to the default, an
isolated 70 second RUN075 SDSS profile with `-t 32` and no RF-accumulator environment flag showed
the first eight store-forced calls at `bins=0` and `rf_store_ms=0.00`, with inner totals of about
37-172 ms. The first profiled chunks still spent about 0.55-0.74 s in initial peel-off, about
1.06-2.01 s in store-forced orchestration, and about 0.74-0.91 s in later peel-off, reaching 11.5%
primary-emission progress before timeout. The next large speedup has to remove that remaining
host-side peel-off and packet-wave orchestration, not the RF table store.

The next RUN075 packet-wave sweep tested larger `SKIRTGPU_BATCH_FIRST_FORCED_CHUNK` values after
the RF store was removed from the hot path. In comparable 70 second capped profiles, the 65536
default reached 11.5% primary-emission progress, 131072 reached 14.0%, 262144 reached 14.6%, and a
too-large 524288 request fell back to roughly 445k-packet host scheduler waves and reached only
12.9%. Combining 262144 with `SKIRTGPU_COMPACT_ROUND_QUEUE=1` reached 14.3%. The promoted default
is therefore 262144 packets per GPU forced-scattering wave for now. A rebuilt no-override default
profile confirmed `count=262144` batch-profile lines and reached 14.1% in the same capped RUN075
window.

The 262144-packet profile split added after that sweep showed where the remaining photon-shooting
time sits. For RUN075, initial emission peel-off was already fully batched
(`emission_peel_batch_ms` equal to `peel_ms`, `emission_peel_scalar_ms=0`), while later scattering
peel-off was still scalar fallback (`later_peel_scalar_ms` about 2.7-3.5 seconds per profiled wave).
`SKIRTGPU_RESIDENT_HG_SCATTER=1`, `SKIRTGPU_DETECTOR_ACCUMULATOR=1`, lifecycle thread cap 8, the
generic batch scatter-peel path, and the direct/fused HG scatter-peel path were all tested against
the promoted 262144 default and were not promoted because capped RUN075 progress did not improve.

The next scatter-peel diagnostic pass removed two pieces of avoidable wrapper overhead without
changing the promoted default: the generic host batch path now stores compact peel-off packets per
observer instead of allocating a full wave-sized `PhotonPacket` vector, and the direct/fused HG
observer-luminosity plus frame-band detector wrappers reuse CUDA scratch buffers. Both changes
compiled and passed `skirtgpu -G`. The capped RUN075 diagnostics still did not justify promotion:
the compact generic batch path reached 7.5% progress in the 70 second window and kept
`later_peel_batch_ms` around 13-15 seconds, while the scratch-backed direct/fused route reached
about 7.1% and direct subprofiles showed roughly 0.8-2.2 seconds in observer luminosity/extinction
plus 1.1-2.1 seconds in detector accumulation per scatter-peel call. The actionable conclusion is
that the next large speedup needs a fused resident observer-to-detector kernel, or a resident packet
queue that performs scattering peel-off across later rounds without returning to host orchestration.

A first fused observer-to-frame detector accumulator was added after that. It is gated by
`SKIRTGPU_BATCH_SCATTER_PEEL_FRAME_FUSED=1` on top of the existing direct scatter-peel diagnostic
and `SKIRTGPU_FRAME_BAND_DETECTOR_DIRECT=1`. The kernel projects each scattering site, traces the
observer optical depth, evaluates the HG phase function and redshifted band transmission, and
atomically accumulates directly into the GPU frame detector; the host wrapper splits each observer
batch across all CUDA runtimes, so the two RTX 5090s each receive a slice of the same batch. The
kernel has a `skirtgpu -G` parity check against the existing fused HG observed-luminosity and
frame-band detector references. It is functional but still not a promoted path: a 70 second RUN075
diagnostic reached 7.4% primary-emission progress, with the first direct subprofiles reporting
`fused_frame=1` and about 1.8-2.3 seconds per scatter-peel group, while the promoted default 70
second post-patch smoke reached 13.9% with scalar later peel-off around 2.3-3.4 seconds per
262144-packet wave. The standalone GPU observer trace remains slower than the 32-thread CPU scalar
path for this workload; the next large gain still needs a resident packet-wave controller that
avoids returning each later scattering peel-off round to host orchestration.

The same fused frame-accumulator kernel was then reused for direct emission peel-off by passing a
zero-asymmetry HG table, making the phase factor exactly one. This path is gated by
`SKIRTGPU_EMISSION_PEEL_FRAME_FUSED=1` plus `SKIRTGPU_FRAME_BAND_DETECTOR_DIRECT=1` and fuses
observer extinction, projection, band transmission, and detector accumulation for the initial
emission peel-off. It compiled, passed `skirtgpu -G`, and a 70 second RUN075 diagnostic reached
14.3% primary-emission progress. The first profiled waves showed `emission_peel_batch_ms` values
from about 0.4 to 1.9 seconds, compared with roughly 1.9 to 2.4 seconds in the post-patch default
smoke, but later scalar scattering peel-off still cost about 2.2 to 4.1 seconds per wave. A full
RUN000 smoke did not justify promotion: the post-patch default completed in 43.5 seconds
(28.7 seconds primary, 9.7 seconds secondary), while the fused-emission diagnostic completed in
44.0 seconds (28.8 seconds primary, 10.0 seconds secondary). SED differences stayed below
5e-5 relative. FITS per-band flux sums differed by at most about 1.3e-3 relative, and the large
pixel-by-pixel image scatter matched a default-vs-default repeat baseline, so it is consistent
with run-to-run Monte Carlo image noise rather than a fused-path-only regression. The path remains
available as an opt-in diagnostic, but it is not promoted by default because it does not improve
RUN000 timing and the later scattering peel-off bottleneck still dominates larger runs.

The secondary-source batch launch patch compiled, passed `skirtgpu -G`, and completed
`RUNs_snap_000_sdss_broadband.ski` with `skirtgpu -g -t 32` in 42.0 seconds on 20 June 2026:
primary emission took 27.6 seconds, secondary emission took 9.6 seconds, and final output completed
normally. The preceding 262144 default RUN000 validation completed in about 44.6 seconds with
secondary emission at about 10.4 seconds. This is a useful secondary photon-shooting cleanup, but
the large speedup still requires moving later scattering peel-off and packet-wave orchestration out
of the scalar host path. After the scatter-peel diagnostic scratch-buffer changes, the same RUN000
smoke completed again in 43.4 seconds with primary emission at 28.7 seconds and secondary emission
at 9.8 seconds, confirming the promoted default path still completes final FITS and SED output.

The 4 September 2026 most-massive-TNG50 test exposed two large-source failures that smaller
benchmarks could not reach. Snapshot 89, subfind 0 has 598,527 imported stellar particles and
6,573 source wavelengths, requiring about 29.3 GiB per GPU for float PDF and CDF tables. Oversized
imported-source PDFs are now packed as bfloat16 while CDFs remain float, reducing these tables to
about 22.0 GiB without reducing the PDF exponent range. The runtime loader also now prefers the
64-bit CUDA `_v2` memory symbols: resolving the legacy symbol first had silently truncated every
allocation above 4 GiB. The float and bfloat16 launch paths pass `skirtgpu -G`. A 10-million-primary
plus 10-million-secondary validation used both RTX 5090s, completed photon shooting in 29.4 seconds,
and kept all 20 FITS image bands finite; integrated band fluxes differed by at most 0.52 percent
from GPU transport with CPU source sampling. The requested production test then completed one
billion primary plus one billion secondary packets in 120 seconds of photon shooting (104 seconds
primary and 15.9 seconds secondary), or 128 seconds end to end. Its four `5 x 318 x 318` FITS cubes
were finite and agreed with the 10-million validation to within 0.56 percent in integrated band
flux. The retained 10-million CPU shooting time of 38.6 seconds projects to about 32 times slower
at one billion packets; this is not a direct billion-packet CPU measurement and remains well below
the 1000x production target.

## Remaining parity work

- Extend the resident lifecycle beyond the current single constant-section HG dust medium,
  nonmoving, unpolarized Voronoi, total-only distant Frame/SED scope. Unsupported grids, materials,
  moving media, polarization, local instruments, and component/statistics outputs still use the
  prepared-path or scalar compatibility paths.
- Extend direct source-to-resident initialization beyond imported BC03/MAPPINGS primary sources and
  the current single-dust Voronoi secondary source. Unsupported primary source types, gas emission,
  multiple dust media, shared cell-library entries, and nonlogarithmic wavelength bias still use the
  packed or scalar launch paths.
- Replace the host-controlled resident round loop with a divergence-aware persistent work queue that
  keeps random streams and survivor scheduling on the GPU while preserving statistically equivalent
  histories. Compacted survivors are now reused by subsequent transport rounds; the tested monolithic
  persistent kernel, 32-round asynchronous groups, and cell-binned detector queue were slower and
  remain disabled.
- Keep source PDF/CDF caches resident across emission iterations when their inputs are unchanged.
  Supported primary and dust-secondary kernels already generate launch states directly in the
  lifecycle structure-of-arrays buffers, but spectra are still constructed on the CPU before the
  first launch of each relevant emission segment.
- Extend grid traversal validation and batching beyond the deterministic single-path diagnostics so
  all built-in grid families are exercised under production-scale photon workloads.
- Extend material-property evaluation beyond the dust-only constant-section table path to
  electrons and gas, wavelength interpolation where required, and spatially/state-dependent
  opacities.
- Port scattering phase functions, wavelength shifts, polarization handling, and random streams so
  the actual material-specific scattering event can execute on the device.
- Move live flux recorder accumulation for SEDs, frames, all-sky, perspective, and IFU/datacube
  instruments to device-side reductions with CPU-identical output; final SED/IFU calibration and
  total-output assembly are already GPU-backed.
- Port secondary emission spectra, including equilibrium and stochastic dust emissivity, MAPPINGS
  and gas line emission where applicable; secondary launch-preparation vector operations are
  already GPU-backed for dust and gas sources.
- Add CPU/GPU parity tests for SEDs and datacubes across no-medium, extinction-only, dust emission,
  gas emission, and dust-and-gas emission modes.

## Broadband and multi-camera optimization pass (2026-10-23)

See `Docs/BROADBAND_OPTIMIZATION_20261023.md` for full details and per-stage profiling tables. Key additions:
- **Tier 0**: Cross-wave device scratch caching (`copyResidentScratchCached`), unified Voronoi plane buffer, top-hat/binary-search band transmission, pre-trace spatial/spectral culling (`skirt_detector_prepare_slot`), warp-aggregated active compaction (`__ballot_sync` + `__popc`), Round-0 compaction, FP32 HG phase evaluation, and analog Russian roulette.
- **Tier 1A**: Hybrid 2-level nested observer optical-depth volumes (`SKIRTGPU_DETECTOR_TAU_VOLUME=1`, default on) combining exact local Voronoi cell traversal (`maxExactSteps = 8`, `s >= 1.5 * du`) with $O(1)$ trilinear remaining-column-density lookup.
- **Tier 1B**: Delaunay greedy neighbor walk (`skirt_voronoi_find_cell_walk_float`), vectorized `float4` plane loads and division-free ray-plane intersection, division-free early-exit majorant DDA with periodic mid-ray column pruning, source-boundary wave splitting (`spanPackets`), and single-feeder sequential wave submission.
- **Tier 1C**: Single-launch shared-memory + warp-shuffle fused source spectrum kernels (`source_spectrum_luminosity_rows_fused`, `source_spectrum_build_rows_fused`), precomputed particle center Voronoi hint cells (`imported_particle_source_prepare_hint_cells`), and in-kernel early wavelength culling + pre-emission active compaction.
- **Tier 2**: Voro++ `c_loop_all`-ordered parallel Voronoi verification and parallel $64^3$ search-block construction in `VoronoiMeshSnapshot.cpp`, elimination of redundant HtoD/DtoH calibration round-trips in `FluxRecorder.cpp` (`SKIRTGPU_GPU_CALIBRATE_ARRAY=0` default), and opt-in deterministic direct stellar emission (`SKIRTGPU_DETERMINISTIC_DIRECT_EMISSION=1`).
- **Measured speedups on single RTX 5090**:
  - `snap99_640113_1e8` (5 Frame instruments, $10^8$ packets): wall time `5.7595 s -> 2.6043 s` (**2.21× faster**), primary emission `4.121 s -> 1.230 s` (**3.35× faster**), GPU kernel total `4173.40 ms -> 1183.14 ms` (**3.53× faster**), peak host RAM `980 MB -> 634 MB` (**-35.3%**), max band flux diff `0.083%`.
  - `snap99_640113_cube25` (10 cameras × 25 bands at $800\times800$, $10^8$ packets): wall time `10.4859 s -> 3.8143 s` (**2.75× faster** default) / `3.7255 s` (**2.81× faster** with `SKIRTGPU_DETERMINISTIC_DIRECT_EMISSION=1`), primary emission `8.125 s -> 1.838 s` (**4.42× faster** default) / `1.701 s` (**4.78× faster** deterministic direct), GPU kernel total `7870.24 ms -> 1796.95 ms` (**4.38× faster** default) / `1658.70 ms` (**4.74× faster** deterministic direct).

