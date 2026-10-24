# SKIRTGPU: where the time goes and what a further 100x would take

Assessment date: 2026-10-16. Method: source reading (`SKIRT/core/GpuAcceleration.cpp`,
`MonteCarloSimulation.cpp`, `MediumSystem.cpp`, the GPU kernel headers) plus the logs, profiles
and run records already on disk under `~/SKIRT/benchmarks/` and
`/data1/skirt_runs/snapshot99_restframe_cube_25band_20261003/`. No code was run for this report.
Every number below is either quoted from an existing log or marked as an estimate.

## 1. Summary

- The GPU port is already 400-700x faster than upstream CPU SKIRT in photon shooting on small,
  thin, ExtinctionOnly galaxies, and 5-100x on dusty or dust-emission workloads. It is a faithful
  GPU translation of the CPU algorithm: one thread per packet walks Voronoi faces exactly as the
  CPU does, and every scattering event traces one exact extinction ray per observer.
- That design is now within a factor of about 2 of its hardware ceiling on small meshes. Kernel
  tuning alone cannot give 100x. Measured GPU time per observer ray is 0.7 ns on a 34k-cell mesh,
  1.6 ns on a 265k-cell mesh and 316 ns on a 1.15M-cell dusty mesh. Cost per event grows with
  mesh size because the face walk is a chain of dependent, scattered memory reads.
- A further 100x therefore has to come from three places at once: (a) algorithms whose cost per
  event does not depend on mesh size (per-observer column-density volumes for peel-off, majorant
  tracking for transport, mixture sampling for source spectra); (b) removing the fixed costs that
  now dominate small galaxies (tessellation, text parsing, per-process CUDA setup, output); and
  (c) estimator changes that need fewer packets for the same image noise (deterministic direct
  light, Russian roulette). Memory goes down, not up: the proposal removes the per-particle
  spectral tables (2-29 GB), the duplicate plane buffers (9.6 GB on halo 0) and the per-observer
  plane caches (up to 60 GB requested on halo 0).
- Honest targets per workload are in section 7. Wall-clock 100x on the smallest production
  galaxy (6.4 s today) is not reachable because 0.6 GB of FITS output and process start-up alone
  cost about 0.5 s. Time-to-equal-noise 100x is plausible on median and large galaxies, and 10-30x
  end-to-end on massive galaxies and dust-emission runs.

## 2. Where SKIRTGPU stands against CPU SKIRT

| Workload | CPU upstream (32 threads) | GPU (2x RTX 5090) | Ratio | Source |
| --- | ---: | ---: | ---: | --- |
| 99_640113, 265k cells, 5 cameras, primary shooting | 145 s for 1e7 packets | 2.2 s for 1e8 packets | ~660x per packet | `skirtgpu_perf_20260930/snap99_640113{,_1e8}` |
| Same, whole run | 148 s (1e7) | 3.4 s (1e8) | ~435x extrapolated | same |
| 746970 broadband, 536.9M packets, shooting | 1458 s | 4.43 s | 329x | `Docs/SKIRTGPU.md` |
| z=1 JWST, 1.15M cells, stochastic dust, whole run | 925 s | 181 s | 5.1x | `Docs/CPU_REFERENCE_20260909.md` |
| Viewer AREPO-095, six self-absorption iterations | 4868 s | 49.9 s | 97x | `Docs/THROUGHPUT_20260930.md` |

The live production batch (25-band rest-frame cubes, 1.7e8 packets, ten 800-928 px cameras,
3120 galaxies completed) is the workload a 100x target must be measured against:

| Cell count | Galaxies | Mean particles | Mean setup | Mean primary | Share of primary time |
| --- | ---: | ---: | ---: | ---: | ---: |
| < 100k | 545 | 59k | 1.1 s | 7.1 s | 7% |
| 100k - 1M | 2010 | 366k | 2.9 s | 13.3 s | 50% |
| 1M - 5M | 402 | 4.7M | 14.3 s | 40.7 s | 31% |
| > 5M | 53 | 30.4M | 74.7 s | 121.0 s | 12% |

Totals over the batch: simulation 78,064 s (median 16 s, p99 205 s, max 412 s), of which setup
16,064 s and primary emission 53,317 s; Python export another 31,516 s. Halo 0 alone: 80.5M
star particles, 19.4M cells, setup 155 s, primary 236 s, 48.7 GB host peak, 428 s export.

## 3. What the production GPU photon cycle actually does

Per wave (one `ChunkMaker` chunk, 1e8/48 to 1.7e8/48 packets with the six primary feeders,
`MonteCarloSimulation.cpp:71-92`, `ChunkMaker.cpp:17`):

1. Host converts all cell sites and densities to normalized FP32 and recomputes block maximum
   densities, every wave (`GpuAcceleration.cpp:11380-11470`), then takes the per-device mutex and
   holds it until the wave finishes (`GpuAcceleration.cpp:11473`). Three feeders queue per device.
2. Uploads FP32 sites and densities again (`:11832-11837`), rebuilds the per-observer compacted
   detector planes and the "half" neighbour planes with two kernels (`:11906-11920`), although the
   grid and the observers never change between waves. Since FP16 planes were dropped, the half
   planes are a full-size duplicate of the FP32 plane cache.
3. Source kernel samples wavelength, position and direction per packet from a per-particle
   PDF/CDF window (`:7468`, window logic `:17656`).
4. Emission detector: one exact Voronoi face walk per packet per observer to the box edge, FP64
   band interpolation for up to every band twice (accept test `:3829`, accumulate `:5260`), FP64
   atomics into frame accumulators.
5. Up to 23 rounds of: analog transport step (one thread per packet walks faces until the sampled
   optical depth or escape, `:6372`), survivor compaction with a host synchronisation and a
   4-byte read-back (`:12334`), scattering detector (again one exact ray per observer per active
   packet), Henyey-Greenstein scatter kernel. Five launches and one sync per round.

Measured per-wave buckets, 99_640113 at 1e8 packets, 6.25M-packet waves
(`snap99_640113_1e8/cand5_prof2/console.txt`):

| Bucket | Per wave | What it contains |
| --- | ---: | --- |
| "setup" | 10 - 210 ms | mostly waiting for the device mutex |
| emission | 43 - 57 ms | source init, plane prep, emission detector (31M rays) |
| transport | 81 - 139 ms | 23 analog steps, first one dominant |
| detector | 29 - 95 ms | 5 - 45M scattering peel-off rays |
| compaction + scatter | ~1.5 ms | 23 rounds |

Sixteen waves sum to about 3.3 s of GPU work on two devices against 2.2 s wall, so each GPU is
busy roughly 75% of the primary phase; the sampled utilisation over the whole 3.9 s process is
48-52% (`cand13_1/run.json`).

The same profile on the z=1 JWST model (1.15M cells, 175k-packet waves,
`z1_jwst_20260905/profile_seed_check/console.txt`) shows the design breaking down: 3.5M packets
produced 198M scattering rays (14 scatterings per packet at the 1e-4 weight cut), transport took
15.1 s and detectors 62.5 s. That is 316 ns per ray and 290 ns per transport step, 100-200x the
small-mesh figures, with a further 165 s of summed mutex waiting across waves.

## 4. Why the current kernels cannot be made 100x faster

Cost model of the exact face walk, per crossing: load the cell's neighbour range, then for each
neighbour a 16-byte plane and a 4-byte index (the detector cache stores forward-facing planes
only, about 8 per crossing). That is roughly 160 bytes of cache traffic per crossing and a chain
of dependent loads. For the 265k-cell galaxy, 31M emission rays times about 25 crossings is
about 125 GB of L2 traffic per wave. At an L2 rate of order 5 TB/s that is about 25 ms; the
measured bucket is 43-57 ms and includes other kernels. The emission detector is thus within
about 2x of the bandwidth ceiling of this algorithm while the mesh fits in L2.

Once the mesh does not fit (planes alone are 276 MB on the JWST mesh, 4.8 GB on halo 0), each
crossing becomes a DRAM round trip, scattering sites sit deep in dense regions with hundreds of
small cells, and the per-ray cost rises two orders of magnitude. No amount of block-size tuning
changes that. The per-observer plane cache that makes the small case fast is also the memory
problem: it needs (4 N_cells + 20 N_faces) bytes per camera, 60 GB for ten cameras on halo 0,
which is why those runs fall back to shared planes and lose another 19%.

Second ceiling: FP64. The RTX 5090 issues FP64 at 1/64 of FP32. Per detector ray the kernel does
FP64 projection, HG phase, redshift, and up to 2 x 25 band interpolations with binary searches;
the transport step does an FP64 `log1p` per sample. Per ray that is of order 100-200 FP64
operations, i.e. 6k-13k FP32 issue slots, comparable to the traversal itself.

Third: host orchestration. One wave per device at a time, a host sync per round, 23 rounds per
wave regardless of how few packets survive, and plane/site re-preparation per wave. This is
10-30% on large waves and dominant on small waves (viewer and secondary passes).

Conclusion: engineering within the present design is worth about 2-3x on shooting. The rest must
come from changing what is computed.

## 5. The plan

### Tier 0: engineering, same estimator, no new science (2-4 weeks)

1. **FP32 detector math.** Projection, pixel index, band interpolation, HG phase in FP32; keep
   only the final accumulate in 64-bit. Expected detector buckets -30 to -50%.
2. **Disjoint-band fast path.** For `ConfigurableBandWavelengthGrid` with disjoint bands (the
   25 top-hats, SDSS/DECam), one binary search over 26 edges replaces up to 50 band evaluations
   per ray. Expected -20 to -40% on the cube workload's detector time.
3. **Cache per-grid state across waves.** Key the FP32 sites, densities, block maxima, detector
   planes and half planes by (grid, observer set, medium generation) exactly like
   `ensureVoronoiPlanesOnDevice` already does (`:17148`). Drop the half-plane buffer entirely
   (identical to the FP32 planes now that `SKIRT_FP32_PLANES` is the default): -4.8 GB on halo 0
   and no 5 GB of writes per wave.
4. **More than one wave in flight per device.** Give each feeder its own stream and scratch set
   instead of one mutex per device; replace the per-round host sync and read-back with a
   device-side round driver (a small persistent kernel that launches or gates the next round from
   the device-visible active count, or CUDA Graph conditional nodes). Expected: utilisation from
   50-75% to above 90%, 1.3-1.6x on large waves, 2-3x on small-wave phases.
5. **Russian roulette** below 1e-2 of launch weight instead of walking every packet down to 1e-4.
   Unbiased; cuts the 23-round tail. Expected -10 to -30% of shooting on thin galaxies, more on
   dusty ones (14 scatterings per packet on the JWST model).
6. **FP32 luminosity integrals.** Halo 0 spends 39 s in `cdfBatch` integrating 80.5M spectra of
   3994 points in FP64. FP32 is sufficient for a normalisation and runs 32-64x faster here.
7. **Binary inputs.** Halo 0 spends 15 s parsing 5.6 GB of text. A binary particle/site reader
   (npy or HDF5 from the exporter, which already holds the arrays) brings that under 1 s.
8. **CUBIN instead of PTX in the kernel cache**, so the driver does not JIT the 26k-line module in
   every process. Needs a measurement; worth up to about 1 s per galaxy.

Expected after Tier 0: small galaxies 1.5-2x end-to-end, large galaxies about 2x.

### Tier 1: mesh-size-independent kernels, same physics (1-3 months)

**A. Per-observer remaining-column-density volumes for peel-off.** For distant instruments all
observer rays are parallel and the dust mix is a single constant-section medium, so
tau(x, lambda) = sigma_ext(lambda) x N_obs(x) where N_obs is the column number density from x to
the box edge along the observer direction. Precompute N_obs once per galaxy per observer on a
(u, v, w) grid aligned with the detector (u, v at pixel or 2x pixel resolution, w along the line
of sight), store the remaining column itself (no cancellation), FP16, compacted to the depth
range where dust exists. Peel-off then costs one trilinear texture fetch per ray instead of a
face walk, independent of cell count. Build cost: one DDA pass per observer over the volume using
point location (B below), of order 10 ms each.

Memory: 800 x 800 x ~256 occupied depth samples x 2 bytes is about 330 MB per camera, 3.3 GB for
ten cameras at 800 px, 4.4 GB at 928 px; this replaces the 1.1 GB-per-camera plane caches on
halo 0. Bias: within one voxel exp(-sigma<N>) >= <exp(-sigma N)>, so clumped sub-voxel dust is
under-attenuated. Control it with 2x oversampling in (u, v) where the per-voxel density variance
is high, and validate by tracing 1% of rays exactly in every wave and comparing. Expected:
detector cost per ray from 1.6-316 ns to about 1-2 ns.

**B. Majorant (Woodcock/delta) tracking with exact point location for transport.** Replace the
face walk with free-path sampling against per-block majorant densities (the 64^3 block maxima
already exist; refine to 128^3-256^3 where contrast is high, 8-64 MB) and accept a tentative
collision with probability rho(cell)/rho_max(block). The only geometric primitive left is "which
cell contains x". Make that exact and O(k): a cell-id hint texture (512^3 x 4 B = 0.5 GB, or
sparse bricks), the k = 16-24 nearest sites per cell (1.2-1.9 GB on halo 0), and the test
|x - s_hint| <= d_k/2 which proves the hint is the true nearest site; fall back to the block search
otherwise. Escaping packets in the thin outskirts become almost free because the majorant is near
zero there. Expected transport per step from 10-290 ns to 5-20 ns, mesh-size independent.

Caveat for dust-emission runs: the radiation field uses the track-length estimator, which needs
exact segment lengths per cell. With delta tracking use the null-collision estimator (deposit
L/sigma_max at each tentative collision) and verify its noise against the track-length result
on the viewer model, or keep the existing exact-walk kernel for the radiation-field-storing
primary pass only.

**C. Node-mixture SED sampling, no per-particle tables.** The BPASS table is interpolated in
log on (Z, t), so a particle's SED is a weighted geometric mean of four node spectra. Precompute
one CDF per table node once per family (tens of MB), sample a node with probability
weight x node luminosity, sample lambda from that node's CDF, and weight by
SED_particle(lambda)/mixture(lambda), which the AM-GM inequality bounds by 1. This removes the
per-particle PDF/CDF tables (2 GiB window today, 22-29 GB requested on the most massive source),
the window rebuilds (halo 0 draws about two packets per particle, so it builds a 3994-point CDF
to use it twice), and most of the 39 s luminosity pass. Apply a hybrid policy: build exact tables
only for particles receiving more than ~1000 packets.

**D. 64-bit fixed-point accumulators** for frames, SEDs and the radiation field (native
`atomicAdd` on `unsigned long long`, deterministic and order-independent, with a per-wave scale
chosen from the total luminosity). Same memory as FP64, faster atomics, bit-reproducible output.

**E. Tessellation-free setup.** With A, B and C in place nothing on the GPU needs Voronoi faces or
neighbour lists. Import sites plus densities (the exporter has the AREPO gas density) instead of
masses that need voro++ volumes, and build the block grid, hint texture and kNN lists on the GPU
(radix sort plus nearest-site queries, under a second at halo-0 scale). Halo 0 setup 155 s drops
to an estimated 15-25 s with binary inputs; small galaxies from 1.1 s to about 0.3 s. Probes that
need cell volumes can estimate them from the hint texture or keep an opt-in voro++ path.

Expected after Tier 1: shooting 3-5x on small galaxies (now dominated by source sampling and
per-event fixed costs), 10-30x on million-cell and dusty meshes; setup 5-10x on large galaxies.

### Tier 2: estimator changes (need science sign-off, 1-2 months)

1. **Deterministic direct light.** The direct image is sum over particles of
   L(lambda) exp(-sigma(lambda) N_obs) splatted with the smoothing kernel. Trace one exact column
   per particle per observer (805M short rays on halo 0, seconds), integrate each particle's SED
   against the bands with that column, splat. No Monte Carlo noise in the dominant image
   component, and the emission detector (30-40% of today's GPU time) disappears. Packets are then
   needed only for scattered light, so the packet count for a given total-image noise drops by
   an amount to be measured with `tools/estimate_jwst_packet_noise.py`; 2-10x is the plausible
   range depending on the scattered fraction.
2. **Weighted delta tracking** (splitting a peel-off branch at tentative collisions) to recover
   the variance reduction that forced scattering gave the CPU and the GPU gave up when it moved
   to analog transport. This matters most in thin regions where few packets scatter at all.
3. **Scattering-order cap** for broadband images with a measured bias, optional.
4. **Service mode.** One resident worker per GPU that keeps the CUDA context, module, SED node
   tables and band tables loaded, accepts galaxies over a socket, overlaps staging and FITS
   writing of galaxy N+1 with rendering of N, and runs two or three galaxies per GPU
   concurrently so small waves never leave a device idle.

## 6. Memory budget

Per GPU, halo-0 scale (19.4M cells, 80.5M particles, 10 cameras at 928 px, 25 bands):

| Item | Today | Proposed |
| --- | ---: | ---: |
| FP32 planes + duplicate half planes + neighbour index | 4.8 + 4.8 + 1.2 GB | 0 |
| Per-observer detector planes | 60 GB requested, falls back | 0 |
| Peel-off column volumes (FP16, compacted) | 0 | ~4.4 GB |
| Cell-id hint texture + kNN lists + block majorants | 0 | 0.5-1.8 + 1.2-1.9 + 0.06 GB |
| Sites, densities | 0.3 GB | 0.3 GB |
| Per-particle PDF/CDF window | 2 GiB (22-29 GB requested on the largest source) | tens of MB of node CDFs |
| Frame accumulators | 1.7 GB FP64 | 1.7 GB fixed-point |
| Resident packet state per wave | ~140 B/packet | ~100 B/packet (FP32 directions, no random scratch) |

Host: the 48.7 GB peak on halo 0 is text parsing plus voro++ plus host copies; binary input and
tessellation-free setup should bring it under 10 GB. JWST-class radiation fields (1.15M cells x
400 bins x 8 B = 3.7 GB per GPU) should be compacted to dust-bearing cells.

## 7. What 100x means per workload

| Workload | Today | Tier 0 + 1 (est.) | + Tier 2 (est.) | Hard floor |
| --- | ---: | ---: | ---: | --- |
| Smallest cube galaxy (41.7k cells) | 6.4 s | ~1.5 s | ~0.5 s | 0.3-0.5 s: 0.6 GB FITS write, process start |
| Median cube galaxy | 16 s | ~2-3 s | ~1 s | output and export |
| Halo 0 (19.4M cells) | 396 s | ~40-50 s | ~30 s | input read, 10 x 928^2 x 25 output |
| z=1 JWST dust emission | 181 s | ~30-40 s | ~20-30 s | self-absorption iterations |
| Node throughput, cube batch | ~5000 galaxies/day | ~30-50k/day | ~50-80k/day | writing 0.6 GB per galaxy: 100x = 3 GB/s sustained |

Wall-clock 100x on a single small galaxy is not available; the fixed costs are already a third of
its runtime. Time-to-equal-noise 100x on median galaxies is plausible when Tier 1 removes the
mesh-bound cost, Tier 2 removes direct-light noise and the packet count follows, and the service
mode removes per-process overhead. For the batch as a whole, output volume becomes the limit
before the GPU does; an FP16 or band-subset cube format should be decided alongside this work.

## 8. Risks and validation

- Column volumes: sub-voxel clumping bias. Keep the exact walk as a 1%-of-rays self-check per
  wave and report the relative difference per band; oversample (u, v) where it exceeds 0.5%.
- Point location exactness: the d_k/2 test is a proof when it passes; log fallback frequency.
- Null-collision radiation-field estimator: compare absorbed luminosities per cell against the
  track-length kernel on the viewer model at equal packets; keep the exact kernel if noise is worse.
- Fixed-point accumulators: choose the scale from total luminosity per wave and check overflow
  headroom (2^63 against 1.7e8 packets is more than 2^35 of resolution at 1e-8 of total flux).
- Deterministic direct light: the splat uses the column at the particle centre; for large
  smoothing lengths sample 4-8 jittered points. Validate against today's output on
  `99_640113` with the two-seed noise tooling, not against pixel identity.
- The GPU already differs from the CPU estimator (analog versus forced scattering; pixel L2
  differences of 5-16% at equal packets are documented). Every step above should be benchmarked
  as time to a stated image noise, never as time at equal packet count.
- Code shape: `GpuAcceleration.cpp` is 26k lines, 87 kernels and about 60 environment flags with
  several disabled experiments. Tier 1 is best built as a separate compact resident pipeline
  (source, transport, peel-off, accumulate) behind one switch, with the present path retained
  for parity runs, rather than as more branches in the existing wave loop.

## 9. Suggested order

1. Tier 0 items 1-3 and 6-7 (days each, measurable with `SKIRTGPU_PROFILE_LIFECYCLE=1` on
   `snap99_640113_1e8` and on halo 0).
2. Node-mixture sampling (Tier 1 C): unblocks the massive galaxies and frees the most memory.
3. Column volumes for Frame instruments (Tier 1 A) with the 1% exact self-check.
4. Streams, device-side rounds, Russian roulette (Tier 0 items 4-5).
5. Delta tracking plus exact point location (Tier 1 B), then tessellation-free setup (E) and
   fixed-point accumulators (D).
6. Deterministic direct light and service mode (Tier 2), with the output-format decision.

Evidence used: `Docs/SKIRTGPU.md`, `Docs/THROUGHPUT_20260930.md`,
`Docs/OPTICAL_BROADBAND_20260927.md`, `Docs/CPU_REFERENCE_20260909.md`,
`Docs/JWST_Z1_20260909.md`, `~/SKIRT/benchmarks/skirtgpu_perf_20260930/` (production galaxy,
halo 0, viewer runs and lifecycle profiles), `~/SKIRT/benchmarks/z1_jwst_20260905/profile_seed_check/`,
`~/SKIRT/benchmarks/optical_500x_20260927/746970/profile/`, and the 3120 per-galaxy logs and
`complete.json` records of `/data1/skirt_runs/snapshot99_restframe_cube_25band_20261003/`.
