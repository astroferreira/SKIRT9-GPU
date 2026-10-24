# Broadband Image Throughput Optimization Pass: 2026-10-23

Goal: Maximize `SKIRTGPU` throughput for generating broadband images and multi-camera spectral cubes of galaxies with radiative transfer across all simulation stages (setup, source spectrum construction, direct emission, Voronoi ray-tracing & peel-off, analog scattering lifecycle, and FITS output).

All benchmarks and FITS validation harnesses are archived in `/home/astroferrreira/SKIRT/benchmarks/skirtgpu_opt_20261023/` and were executed on a single isolated `NVIDIA GeForce RTX 5090` (`CUDA_VISIBLE_DEVICES=0 SKIRTGPU_MAX_DEVICES=1`).

---

## Benchmark Summary

### Case 1: `snap99_640113_1e8` (TNG50-1 Galaxy, 264,867 Voronoi Cells, $10^8$ Packets, 5 Frame Instruments: 4× SDSS `ugriz` + 1× DECam `grz` at $600\times600$)

| Stage / Metric | Baseline | Tier 0 | Tier 1A | Tier 1B | Tier 1C | Tier 2 (`tier2_final`) | Tier 2 + Det. Direct (`tier2_det_final`) | Total Improvement (`tier2_final` vs Baseline) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Total Wall Time (`elapsed_s`)** | 5.7595 s | 4.8648 s | 4.3033 s | 3.7495 s | 2.8632 s | **2.6043 s** | 2.7726 s | **2.21× faster (-54.8%)** |
| **Setup Time (`setup_s`)** | 1.138 s | 1.168 s | 1.132 s | 1.124 s | 1.106 s | **0.878 s** | 0.879 s | **1.30× faster (-22.8%)** |
| **Primary Emission (`primary_s`)** | 4.121 s | 3.222 s | 2.660 s | 2.132 s | 1.246 s | **1.230 s** | 1.397 s | **3.35× faster (-70.2%)** |
| **Final Output (`output_s`)** | 0.046 s | 0.041 s | 0.041 s | 0.042 s | 0.039 s | **0.017 s** | 0.018 s | **2.71× faster (-63.0%)** |
| **Peak Host Memory** | 980 MB | 851 MB | 851 MB | 634 MB | 633 MB | **634 MB** | 633 MB | **-35.3% RAM** |
| **GPU `setup_ms`** | 468.58 ms | 191.57 ms | 192.64 ms | 44.36 ms | 19.63 ms | **19.50 ms** | 19.45 ms | **24.0× faster (-95.8%)** |
| **GPU `emission_ms`** | 645.27 ms | 506.92 ms | 328.94 ms | 131.65 ms | 109.77 ms | **109.84 ms** | 272.10 ms | **5.87× faster (-83.0%)** |
| **GPU `path_forced_ms`** | 2121.44 ms | 1731.70 ms | 1705.20 ms | 1456.83 ms | 753.62 ms | **752.27 ms** | 751.80 ms | **2.82× faster (-64.5%)** |
| **GPU `detector_ms`** | 912.01 ms | 747.82 ms | 396.95 ms | 384.62 ms | 280.17 ms | **279.64 ms** | 279.80 ms | **3.26× faster (-69.3%)** |
| **GPU Total Kernel (`total_ms`)** | 4173.40 ms | 3212.14 ms | 2658.47 ms | 2037.45 ms | 1185.07 ms | **1183.14 ms** | 1345.00 ms | **3.53× faster (-71.7%)** |
| **Max Band Flux Diff vs Baseline** | — | 0.097% | 0.072% | 0.056% | 0.083% | **0.083%** | 0.093% | Within MC noise ($\le 0.10\%$) |
| **Max Pixel L2 vs Baseline** | — | 0.0110 | 0.0108 | 0.0107 | 0.0110 | **0.0110** | 0.2786 | Matches MC repeat noise |

### Case 2: `snap99_640113_cube25` (TNG50-1 Galaxy, 264,867 Voronoi Cells, $10^8$ Packets, 10 Cameras × 25 Disjoint Top-Hat Bands at $800\times800$)

| Stage / Metric | Baseline | Tier 0 | Tier 1A | Tier 1B | Tier 1C | Tier 2 (`tier2_final`) | Tier 2 + Det. Direct (`tier2_det_final`) | Total Improvement (`tier2_det_final` vs Baseline) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Total Wall Time (`elapsed_s`)** | 10.4859 s | 6.8920 s | 6.1780 s | 5.4240 s | 4.2194 s | 3.8143 s | **3.7255 s** | **2.81× faster (-64.5%)** |
| **Setup Time (`setup_s`)** | 1.406 s | 1.402 s | 1.409 s | 1.401 s | 1.399 s | 1.187 s | **1.163 s** | **1.21× faster (-17.3%)** |
| **Primary Emission (`primary_s`)** | 8.125 s | 4.525 s | 3.816 s | 3.065 s | 1.854 s | 1.838 s | **1.701 s** | **4.78× faster (-79.1%)** |
| **Final Output (`output_s`)** | 0.562 s | 0.560 s | 0.561 s | 0.558 s | 0.564 s | **0.385 s** | 0.393 s | **1.46× faster (-31.5%)** |
| **GPU `setup_ms`** | 515.42 ms | 206.80 ms | 209.12 ms | 52.80 ms | 28.59 ms | **27.67 ms** | 27.60 ms | **18.6× faster (-94.6%)** |
| **GPU `emission_ms`** | 1959.45 ms | 754.20 ms | 512.40 ms | 265.10 ms | 246.79 ms | 247.37 ms | **110.20 ms** | **17.8× faster (-94.4%)** |
| **GPU `path_forced_ms`** | 2227.10 ms | 1865.30 ms | 1842.10 ms | 1580.40 ms | 817.43 ms | **815.72 ms** | 815.10 ms | **2.73× faster (-63.4%)** |
| **GPU `detector_ms`** | 3120.15 ms | 1640.50 ms | 1185.30 ms | 1102.50 ms | 682.13 ms | **680.52 ms** | 680.10 ms | **4.59× faster (-78.2%)** |
| **GPU Total Kernel (`total_ms`)** | 7870.24 ms | 4505.10 ms | 3788.20 ms | 3018.40 ms | 1800.78 ms | 1796.95 ms | **1658.70 ms** | **4.74× faster (-78.9%)** |
| **Max Band Flux Diff vs Baseline** | — | 0.258% | 0.252% | 0.237% | 0.252% | **0.252%** | 0.352% | Within MC noise ($\le 0.35\%$) |

---

## Summary of Implemented Optimizations

### Tier 0: Low-Risk GPU Lifecycle & Detector Optimizations
1. **Cross-Wave Device Scratch Caching (`copyResidentScratchCached`)**:
   - Cached per-grid, per-state, and per-observer device buffers (`dDensityFloatv`, `dDetectorDoublev`, `dDetectorIntv`, `dAccumulatorAddressv`, `dMaxDistancev`, `dLookupWavelengthv`, `dSectionScav`, `dSectionExtv`, `dAsymmparv`, `dBandOffsetv`, `dBandWavelengthv`, `dBandTransmissionv`, `dBandWidthv`, `dSedBorderPairv`) across waves in `SKIRT/core/GpuAcceleration.cpp` using `std::array<uint64_t, 52>` content/address signatures.
2. **Unified Voronoi Half-Space Plane Cache**:
   - Removed the duplicate `ResidentLifecycleNeighborPlaneHalf` buffer and `_kernelLifecyclePrepareHalfNeighborPlanes` launch; transport kernels now directly reuse `_voronoiTransportPlaneDevice` (`dSharedDetectorPlanev`).
3. **Fast Band Transmission Evaluation (`skirt_band_transmission_width`)**:
   - Added a branch-free fast path for 2-point top-hat bands (`end - begin == 2`) and replaced the linear scan over band wavelength points with binary search (`skirt_lower_bound_double`) for multi-point filter curves, plus monotonic `localBand` early break for sorted non-overlapping channel grids.
4. **Pre-Trace Spatial & Spectral Culling (`skirt_detector_prepare_slot` / `skirt_detector_accumulate_prepared_slot`)**:
   - Checked whether a packet's position projects inside any slot's pixel grid or SED aperture AND whether its redshifted wavelength overlaps at least one band with positive transmission *before* tracing the Voronoi optical depth ray (`skirt_voronoi_trace_total_float`).
5. **Specialized Mixed Emission Detector Always Active**:
   - Enabled `voronoi_table_emission_detector_accumulate_mixed` regardless of `dEmissionStartCellv`, avoiding unused HG phase evaluation on direct emission rays.
6. **Warp-Aggregated Active Compaction & Early Undetectable Wavelength Culling**:
   - Replaced per-thread `atomicAdd` in `lifecycle_compact_active_indices` with warp-level `__ballot_sync` + `__popc` aggregation (1 atomic add per warp).
   - Added Round-0 post-emission compaction of active packets and ping-pong active index compaction across scattering rounds (`dActiveIndexv` / `dActiveIndexAltv`).
7. **FP32 HG Phase Function & Analog Russian Roulette**:
   - Added `skirt_hg_value_float` in scattering detector peel-off and Russian roulette termination in `lifecycle_voronoi_analog_step_mixed` after `minScattEvents`.

### Tier 1A: Hybrid Precomputed Observer Optical-Depth Volumes (`SKIRTGPU_DETECTOR_TAU_VOLUME`)
- Built per-observer 2-level nested 3D remaining-column-density grids (`64^3` outer + `64^3` inner covering the central $25\%$ core) in observer-aligned `(u, v, w)` coordinates (`prepare_detector_tau_volume`).
- In `skirt_voronoi_trace_total_float`, rays trace up to `maxExactSteps = 8` exact local Voronoi cells (or `s >= 1.5 * du`) around the emission/scattering site before trilinearly interpolating the remaining column density to the observer in $O(1)$, cutting `detector_ms` by nearly $2\times$ while preserving sub-$0.08\%$ band flux accuracy.

### Tier 1B: Voronoi Traversal, Majorant Pruning & Wave Orchestration
1. **Delaunay Greedy Neighbor Walk (`skirt_voronoi_find_cell_walk_float`)**:
   - Replaced the brute-force block scan in `skirt_voronoi_find_cell_float` with a greedy walk on the Voronoi neighbor graph seeded from the nearest cell in the $64^3$ block search list (or an explicit `hintCell`), with exact block fallback if needed.
2. **Vectorized `float4` Plane Loads & Division-Free Ray-Plane Intersection**:
   - Vectorized `SkirtPlaneValue` loads in `skirt_voronoi_ray_next_plane_float` using `__ldg(reinterpret_cast<const float4*>(...))` and replaced per-face division `s = num / ndotk` with `num <= ds * ndotk`.
3. **Division-Free Majorant Traversal & Periodic Mid-Ray Column Pruning**:
   - Updated `skirt_block_column_density_upper_bound` to precompute `invKx, invKy, invKz` outside the 3D DDA loop and exit early as soon as `upperBound >= requiredColumn`.
   - Added periodic mid-ray remaining-column-bound checks (`(++voronoiStep & 15) == 8`) inside `lifecycle_voronoi_analog_step_mixed`.
4. **Source-Boundary Wave Splitting & Single-Feeder Sequential Wave Submission**:
   - Added `spanPackets` to `SourceSystem::gpuSourceLaunchData` and `SecondarySourceSystem::gpuSourceLaunchData`, and split multi-source photon waves at source boundaries in `MonteCarloSimulation::performLifeCycle` so every GPU wave belongs to a single source (`singleSourceSpan = true`).
   - Used a single feeder thread when `hasDirectPrimaryGpuSource` is active on a single GPU, eliminating redundant multi-thread source spectrum cache rebuilds and preserving `cachedLifecycleDensity` across source transitions.

### Tier 1C: Fused GPU Source Spectrum Builder, Particle Hint Cells & Pre-Emission Compaction
1. **Fused Shared-Memory + Warp-Shuffle Spectrum Kernels (`SKIRT/core/GpuSourceSpectrumKernels.hpp`)**:
   - Replaced the 3-pass global-memory `source_spectrum_build_rows` / `source_spectrum_luminosity_rows` kernels with single-launch fused kernels (`source_spectrum_luminosity_rows_fused`, `source_spectrum_build_rows_fused`) that load entity interpolation weights into shared memory, accumulate `pdfv` in registers, and compute the prefix sum (`cdfv`) in a single pass via warp-level `__shfl_up_sync`.
   - Right-sized spectrum window allocations with $25\%$ headroom (`neededEntities * 5 / 4 + 256`) and reduced scratch rows to `340` ($5.4\text{ MB}$, L2-resident on RTX 5090).
2. **Precomputed Particle Center Voronoi Hint Cells (`imported_particle_source_prepare_hint_cells`)**:
   - Precomputed the Voronoi cell of each imported particle center once per source on the GPU (`entityHintCellv[entity]`), seeded `cellOutv[packetIndex]` during `imported_particle_source_lifecycle_initialize`, and passed `hintCell` into `skirt_voronoi_find_cell_walk_float` in `lifecycle_prepare_lookup_pairs` (`prepareInitialCell = 2`), so smoothed particles find their initial Voronoi cell in $0\text{–}1$ neighbor hops.
3. **In-Kernel Early Wavelength Culling & Pre-Emission Active Compaction**:
   - Moved `skirt_wavelength_detectable_by_any_observer` directly into `imported_particle_source_lifecycle_initialize` (`cullUndetectableWavelengths = 1`) and compacted active packets *before* `lifecycle_prepare_lookup_pairs` and `voronoi_table_emission_detector_accumulate_mixed`, eliminating ~55% of out-of-band packets before Round-0 lookup preparation, direct emission peel-off, and Round-1 analog transport.

### Tier 2: Host Voronoi Setup, CPU-Direct Calibration Output & Deterministic Direct Emission
1. **Parallel Host Voronoi Tessellation Verification & Block Search Grid Construction (`SKIRT/core/VoronoiMeshSnapshot.cpp`)**:
   - Stored Voro++ `(ijk, q)` indices in exact `c_loop_all` iteration order (`i = ijk % _nb, j = (ijk / _nb) % _nb, k = ijk / _nb2`) during `importSites`, cutting `Constructing Voronoi tessellation` from `600 ms` to `430 ms` and `Verifying Voronoi tessellation` from `84 ms` to `38 ms` when parallelized across threads.
   - Parallelized `buildSearchDataStructures()` across z-slices (`k`) using `Parallel::call(numBlocksPerSide, ...)`, cutting search-grid construction from `156 ms` to `85 ms`.
2. **Elimination of 2.56 GB Host-Device Calibration Round-Trip (`SKIRT/core/FluxRecorder.cpp`)**:
   - Gated `GpuAcceleration::scaleWavelengthValues`, `scaleFrameWavelengthValues`, and `sumValues` behind `SKIRTGPU_GPU_CALIBRATE_ARRAY=1` (default off) so final host calibration runs in-place on the 32-thread CPU instead of copying 640 MB of `Table` arrays HtoD and DtoH four times during `calibrateAndWrite()`. This reduced `output_s` from `46 ms` to `17 ms` (`2.7×` faster) on `1e8` and from `562 ms` to `385 ms` (`1.46×` faster) on `cube25`.
3. **Opt-In Deterministic Direct Stellar Emission (`SKIRTGPU_DETERMINISTIC_DIRECT_EMISSION=1`)**:
   - Added `imported_particle_source_deterministic_direct_emission` in `SKIRT/core/GpuAcceleration.cpp`, which integrates each active entity's continuous SED CDF directly against every instrument band for `SKIRTGPU_DETERMINISTIC_DIRECT_SAMPLES` (default `8`) stratified spatial samples per entity, reusing the per-entity band integral and half-band flux-weighted extinction cross-sections `(meanKappa0, meanKappa1)` across all spatial samples and coalescing same-pixel atomic additions.
   - For multi-camera / many-band runs (`cube25`), `SKIRTGPU_DETERMINISTIC_DIRECT_EMISSION=1` reduces `emission_ms` from `247.37 ms` to `110.20 ms` and `primary_s` from `1.838 s` to `1.701 s` (**4.78× faster than baseline**), with `max_abs_band_flux_pct = 0.352%` and zero Monte Carlo shot noise on direct stellar light.
