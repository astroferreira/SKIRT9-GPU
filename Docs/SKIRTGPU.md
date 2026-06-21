# SKIRTGPU implementation notes

This document tracks the incremental GPU port. The target state is a SKIRTGPU build with
one-to-one CPU parity for galaxy SEDs and IFU datacubes, including stellar, gas and dust sources,
BC03 and MAPPINGS SEDs, dust attenuation, scattering and emission.

## Current GPU coverage

- `skirtgpu` is built next to the regular `skirt` executable.
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
- Secondary emission now uses the same contiguous batch launch entry point as primary emission. The
  default secondary-source implementation preserves scalar behavior, while the dust secondary source
  walks contiguous cell history-index ranges once and reuses the cell mapping for every packet in
  the range. This is still CPU-orchestrated packet initialization, but it removes repeated
  source/cell binary searches from the secondary photon-shooting hot path and provides the hook for
  future GPU dust-emission wavelength sampling.
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
- Forced-scattering rounds can be processed in packet waves after the first
  propagation step. In this mode, material-specific scattering and scattering peel-off remain
  CPU-side, but every subsequent path/tau preparation, radiation-field contribution, and forced
  propagation step is batched again instead of returning each photon packet to scalar lifecycle
  transport.
- Scattering peel-off profiling is split from material scattering in the experimental
  forced-round controller. `SKIRTGPU_PROFILE_BATCH=1` now separates initial emission peel-off from
  later scattering peel-off and also reports whether each used the batch wrapper or scalar fallback.
  The 20 June 2026 262144-packet RUN075 profile showed initial emission peel-off fully on the batch
  observer-extinction path (`emission_peel_scalar_ms=0`), while later scattering peel-off still fell
  back to scalar work (`later_peel_scalar_ms` about 2.7-3.5 seconds per profiled wave). A host-side
  batch wrapper for distant-instrument scattering peel-off is available as a separate opt-in
  diagnostic. For total-only `FrameInstrument` outputs this wrapper can now reduce detector
  `(pixel,wavelength)` additions with a generic GPU key-sum kernel, and its compact peel-packet
  storage avoids constructing a full packet wave per observer. It is still not part of the faster
  RUN075 path because the batch currently adds more host peel-off packet construction and GPU path
  setup overhead than it saves.
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
  avoiding a temporary direction vector. The resulting observed optical depths are stored on the
  peel-off packets before the existing instrument detection code runs, preserving the scalar
  detector accumulation order for SED, image, and IFU/datacube outputs.
- Imported BC03 and MAPPINGS source luminosity setup can evaluate stored-table CDF normalization
  factors in a single GPU batch over all imported entities. The CPU scalar `SEDFamily::cdf()` path
  remains the fallback for unsupported SED families or unavailable GPU runtime.
- Imported BC03 and MAPPINGS primary-source launch preparation can batch sample source wavelengths
  and normalized SED probabilities on the GPU for the history-index range assigned to a source.
  CPU-generated random choices and bias-distribution samples are retained, and unsupported SED
  families fall back to the existing per-entity launch cache.
- Primary `SourceSystem` launch now has a contiguous batch API. The default implementation keeps
  scalar launch behavior, while `ImportedSource` walks source/entity history-index ranges once and
  reuses the existing GPU-batched wavelength preparation. This is still CPU-orchestrated for
  positions, directions, velocities, and `PhotonPacket` state, but it removes one source/entity
  lookup layer from the 65536-packet forced-scattering chunks and provides the compatibility hook
  for a future device-resident imported-source launch kernel.
- Imported-source batches without imported velocities can now convert pre-drawn isotropic launch
  direction randoms to direction vectors with a GPU batch kernel. The CPU still owns the random
  stream and packet initialization, so the kernel is a launch-path primitive rather than a complete
  resident source kernel.
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
- `SKIRTGPU_RESIDENT_FORCED=0`: opt out of the resident Voronoi/table store-radiation-field plus
  forced-propagation pass and use the older prepared-path batch path.
- `SKIRTGPU_RESIDENT_SAMPLE_ON_GPU=0`: keep the resident Voronoi/table pass but sample the forced
  interaction optical depth after a separate GPU total-optical-depth pass. This exact resident
  fallback is useful for debugging; on the current RUN075 profile it was slower than the default
  sampled resident path.
- `SKIRTGPU_BATCH_FIRST_FORCED_CHUNK=<count>`: photon-packet chunk size used by the
  forced-scattering batch. The default is 262144 for production `-g` runs on the current
  host-driven pipeline because it improved RUN075 two-RTX-5090 primary-emission throughput after
  direct RF accumulation became the default. Setting this to 1000 is useful for matched-order
  regression checks against older smoke baselines; setting it back to 65536 is useful for A/B
  comparisons with earlier profiles.
- `SKIRTGPU_LIFECYCLE_THREADS=<count>`: optionally cap CPU feeder threads for GPU-enabled photon
  lifecycle work. The default is uncapped because RUN075 throughput was best with the regular SKIRT
  CPU parallel scheduler. On 20 June 2026 with the 65536 chunk and two balanced RTX 5090 devices,
  4, 8, 12, and 16 lifecycle threads reached 3.0%, 4.8%, 5.9%, and 6.5% primary progress in a
  65 second RUN075 window, still below the uncapped 6.9%.
- `SKIRTGPU_LIMIT_LIFECYCLE_THREADS=0`: ignore `SKIRTGPU_LIFECYCLE_THREADS` even if it is set.
- `SKIRTGPU_BATCH_FORCED_ROUNDS=0`: opt out of batched later forced-scattering rounds after the
  first propagation step. This is enabled by default for production `-g` runs because it improves
  RUN075 throughput, but the larger packet waves change random-stream interleaving across packets.
- `SKIRTGPU_COMPACT_ROUND_QUEUE=1`: opt into a compact host-side survivor queue for later
  forced-scattering rounds. The default keeps the previously benchmarked full-chunk scan. On the
  post-RF-accumulator RUN075 profile it slightly improved some 65536-packet chunk timings but did
  not improve the capped progress, and combining it with the 262144-packet wave was slightly
  slower than the promoted default, so it remains opt-in.
- `SKIRTGPU_BATCH_SCATTER_PEEL=1`: opt into the diagnostic host-side scattering peel-off batch for
  distant instruments during batched forced rounds. Total-only frame detectors use a GPU key-sum
  reduction for compact flux accumulation in this mode. It is off by default because the 20 June
  2026 RUN075 diagnostics were slower than the promoted scalar-peel batched-round path. After the
  shared-direction observer-extinction cleanup and the 262144-packet default promotion, capped
  profile retries still showed worse throughput: the generic host batch scatter-peel path reached
  only 7.1%, and the direct HG path combined with the direct frame-band accumulator spent about
  11-18 seconds in `later_peel_batch_ms` per profiled wave. This remains diagnostic.
- `SKIRTGPU_BATCH_SCATTER_PEEL_FUSED=1`: when combined with
  `SKIRTGPU_BATCH_SCATTER_PEEL_DIRECT=1`, opt into the fused Voronoi/table HG observer-luminosity
  kernel. This kernel computes the HG phase factor and observer extinction in one GPU pass and
  writes luminosities that are ready for total-frame detection. It remains diagnostic: with the
  65536 chunk on the full RUN075 file, 65 second progress was 2.2% for direct HG scatter peel-off
  and 2.4% for fused direct, versus 6.9% for the promoted default route. A 20 June 2026 retry of
  fused direct scattering peel-off after direct RF accumulation was promoted still reached only
  4.0% in the 70 second capped-profile window and showed `later_peel_ms` around 5.4-8.0 seconds.
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
  detector key/value preparation. This avoids host `LockFree::add()` for total-only distant
  frame outputs, but it is still diagnostic while the photon-package loop remains host
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
the CPU SKIRT version. Current scalar and setup kernels are stepping stones toward that target;
meeting it requires batched photon propagation with resident GPU grid, material, source, and
instrument state so the Monte Carlo loop is not driven packet-by-packet from the CPU.

The current RUN075 checkpoints in this branch are stable but not a 100x or 1000x result. On
20 June 2026, a single-process two-GPU bounded run on
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

## Remaining parity work

- Batch photon packets so path generation, optical-depth accumulation, radiation-field storage,
  interaction-point selection, scattering peel-off, local-perspective instrument peel-off, and later
  propagation rounds can run with high GPU occupancy.
- Replace host-side `SpatialGridPath` reconstruction with a fused resident
  traversal-plus-optical-depth kernel. The GPU now uses Voronoi search-grid block lists for
  initial-cell lookup and can fuse dust-table optical-depth accumulation into compact path output,
  but the path is still copied back to the CPU and used as the compatibility boundary for
  radiation-field preparation, forced-propagation, detector peel-off, and later lifecycle stages.
- Replace the experimental batched-round controller with a device-resident packet queue that keeps
  packet state, random streams, path data, forced propagation, and termination decisions on the GPU
  while preserving CPU-identical histories.
- Move the imported-source launch cache into the future device-resident photon-packet pipeline;
  BC03/MAPPINGS wavelength sampling and isotropic direction conversion have batch GPU primitives,
  and packet path-segment buffers are no longer eagerly allocated, but positions, packet state, and
  propagation are still orchestrated by the CPU.
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
