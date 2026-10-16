# Image throughput pass: 2026-09-30

Goal: thousands of SKIRTGPU images per day on the two-RTX-5090 node. Two real
workloads were profiled: the snapshot-99 broadband batch (8,252 TNG50 galaxies,
ExtinctionOnly, 1e8 packets, five Frame instruments) and the Observatory viewer's
dust-emission renders (stochastic DL07 dust, iterated self-absorption).

Evidence, inputs, harnesses and every run directory are under
`/home/astroferrreira/SKIRT/benchmarks/skirtgpu_perf_20260930/`. All timings are
external wall times with the CPU reference process paused (SIGSTOP) while GPU runs
were timed.

## Results

| Workload | Before | After | Notes |
| --- | ---: | ---: | --- |
| Production galaxy 99_640113, 1e8 packets | ≈700 s* | 4.0 s | default wave size; 13.4 s with the batch's 131072 override |
| Same, 3 concurrent jobs | — | 3.28 s/galaxy | 26,300 galaxies/day |
| Halo 0 (65M stars), production-style SKI | ≈322 s** | 75 s | text parse 53→10 s, luminosities 43→16 s, primary 69→33 s |
| Viewer AREPO-095, 1e7 packets, old viewer env | 186.6 s | 65.3 s | GPU memory 9.0 → 3.7 GB per device |
| Same with the updated viewer environment | — | 42.1 s | 4.4x |
| CFIS 10-camera galaxies (smoke run, 4 galaxies) | — | 7–21 s | simulation only; export 4–14 s |

\* Batch logs show 582–972 s per galaxy at 1e8; 99_640113 itself was interrupted.
\** The earlier halo-0 run used a dust-emission SKI; the production-style SKI with the
old binary was not rerun. Its 184 s first measurement already included the observer-slot fix.

## Changes

1. **Duplicate viewing directions no longer disable the resident lifecycle.**
   The production SKI has `xy` (SDSS) and `decals_xy` (DECam) at identical angles.
   Observer assignment merged them into one slot, found two Frames, and rejected the
   whole simulation, so every packet ran through the host-orchestrated fallback.
   `assignResidentObserverSlots()` gives a further Frame/SED at a shared viewpoint its
   own slot. SDSS benchmarks without duplicates are bit-identical and equally fast.
2. **Stochastic dust kernel rewritten.** One warp per cell; blocks loop over cells, so
   no per-cell scratch matrix (formerly a 1500² FP64 triangle, 4.4 GB per GPU at 512
   cells). The cumulative-matrix solve is replaced by the equivalent recurrence
   `Σ_{j<i} A'(i,j)P_j = Σ_{f≥i} Q(f)`, with column prefetch into registers. Heating
   tables are column-major. Default batch 16,384 cells (`SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS`
   now accepts up to 1,048,576). Per pass 15.8 s → 3.6 s; CPU validation error ≤ 5e-14.
3. **FP32 Voronoi plane caches.** The mixed-precision lifecycle stored face planes in
   FP16, displacing faces by ~1e-4 of the box (tens of pc for 150 kpc boxes). This biased
   fluxes by about −0.5% on small galaxies and raised dust self-absorption by 3–4%.
   `SkirtPlaneValue` (kernel macro `SKIRT_FP32_PLANES`, default 1) now stores FP32
   at no measurable cost. Per-band fluxes now match a full-FP64 run, and absorbed
   luminosities match upstream CPU SKIRT: primary −0.08%, iteration-1 secondary −0.26%.
4. **Resident self-absorption passes.** Radiation-field-only passes (no peel-off) run
   resident with a detector-less "null observer" when packets are initialized on the GPU
   (`SKIRTGPU_RESIDENT_RF_ONLY=0` disables). Packed/host-launched variants stay peel-off
   only; with CPU-launched dust packets that combination took 489 s per iteration.
5. **Fast text parsing.** `TextInFile::readRow` uses `std::from_chars` instead of a
   `stringstream` per line (5.3x on 5.6 GB of star particles, bit-identical values). Error
   behaviour is kept: missing values and non-numeric tokens are errors, NaN is accepted,
   infinities and overflow are rejected, a leading `+` is allowed.
6. **Cheaper SED integrals.** Power-law segment areas use `(x1·p1 − x0·p0)/(1+α)` in the
   general branch, which is algebraically identical to the `gln` form, instead of an FP64
   `pow`. Validated against CPU luminosities at 6.5e-14.
7. **Source spectrum windows reuse the device SED table** (and scratch) instead of
   re-uploading and re-logging it for every cache window.
8. **Same-direction observer slots share one traced ray** in the mixed emission and
   scattering detector kernels (`SKIRTGPU_SHARE_OBSERVER_TRACES=0` disables). Outputs
   are bit-identical; throughput +4% for the production SKI.

9. **Source-spectrum windows follow each launch** instead of fixed aligned blocks: a launch reuses the
   cached rows when they contain its first entity, otherwise builds exactly the entities it needs, and
   the cache buffers are high-water. Concurrent feeders no longer rebuild unneeded rows.
10. **Log-space spectrum integrals.** The PDF builder keeps the interpolated log-PDF, so the power-law
    segment integrals use `α = (y₁ − y₀)/ln(x₁/x₀)` with a precomputed log-ratio table and need no
    `log` calls. Halo 0: luminosities 28→16 s, primary 47→33 s, CPU luminosity checks ≤ 1.5e-13.
    `SKIRTGPU_SOURCE_SPECTRA_FP32=1` additionally evaluates cache exponentials in FP32 (opt-in, ~6% more).
11. **Parallel stochastic-dust heating tables.** The per-grain heating triangles were filled serially
    during setup (4.4 s for DL07 with 60 grains); rows are now distributed like the cooling rates.
12. **Adaptive dust batches.** About 1/16 of the emitting cells per launch (2048–16384) balances both
    GPUs: 3.3 s per pass instead of 3.6–4.2 s.

`SKIRTGPU_DUMP_DUST_BATCH=<prefix>` writes four complete stochastic-dust kernel batches
for the standalone harness `dust_harness.py`.

## Deployment (2026-10-01)

- `release/SKIRT/main/skirtgpu` replaced; previous binaries kept as `skirtgpu-a78d07-20260930` and
  `skirtgpu-e6de7d-20261001`.
- `run_snapshot99_broadband_batch.py` (shared by the CFIS runner) no longer sets
  `SKIRTGPU_BATCH_FIRST_FORCED_CHUNK=131072`.
- Viewer: `SKIRTGPU_SOURCE_LAUNCH=0` removed; the dust batch override is only passed when
  `SKIRT_VIEWER_DUST_BATCH_CELLS` is set. Viewer tests pass.
- Prepared batch plans pin the binary SHA-256, so a fresh CFIS plan was prepared against the deployed
  binary. Older plans must be re-prepared before resuming.
- The CFIS runner accepts `--shard I/N` for `run` and `stage`: shard I processes every N-th selected
  galaxy. Shards share the output lock (an unsharded run still excludes them), write
  `status.shardIofN.json`, and keep the per-galaxy checkpoint/resume logic unchanged. Start all
  shards together, because each checks for 24 GiB of free GPU memory at startup:
  `./run_batch.sh --shard 0/2` and `./run_batch.sh --shard 1/2` in separate tmux windows.
- Full batch on the 15 TB HDD, restarted 2026-10-01 14:12 local, most massive galaxies first, as a
  single runner (`run_batch.sh --descending`; tmux `skirtgpu-cfis-full`; `start_tmux.sh` resumes):
  `/data1/skirt_runs/snapshot99_cfis_mergers_10views_1e8_full_20261001c`. Earlier attempts
  (`..._full_20261001` with 246 galaxies, `..._full_20261001b` with 4) are kept for reference.
- `--descending` reverses the plan order before `--limit`/`--shard`. Cutouts above 1 GiB take
  `.heavy.lock`, so concurrent shards never process two such galaxies at once. It runs as two shards in
  tmux sessions `skirtgpu-cfis-full-shard0` and `skirtgpu-cfis-full-shard1`; `start_sharded_tmux.sh`
  in that directory resumes both. Raw cutouts are borrowed from the broadband plan's cache on
  `/home` and released after each verified galaxy.
- RGB delivery: `upload_rgb_zips.py` in the plan directory (tmux `skirtgpu-cfis-rgb-upload`) zips the
  grz PNGs of newly completed galaxies (300+ per part, every 30 min, final remainder at the end) into
  `gdrive:SKIRTGPU/snapshot99_cfis_mergers_10views_1e8_full_20261001_rgb`. Per-file uploads were
  throttled to about 4 PNGs per minute by the shared rclone OAuth client; a personal Drive client ID
  would allow individual files again.

## Pipeline outside SKIRT (illustris_to_stellarmap)

With SKIRT at seconds per galaxy, the Python export became the bottleneck. In the v3 CFIS plan,
exports took 94–356 s for large cutouts against 18–56 s of simulation. Profiling a 258 MB cutout
(7.8M stars, 131.9 s export) showed:

- 80 s in astropy `lookback_time`, which integrates every particle with `quad` (about 7 µs each).
  Stars formed in one timestep share a float32 scale factor (80M stars have about 1.7M distinct
  values), so `stellar_ages` now evaluates each distinct value once. The ages are bit-identical.
- 22.5 s in the k-NN smoothing-length query (`cKDTree`, 8 workers); it now uses all cores.
- 18 s in `np.savetxt('%.17g')`; large tables are formatted in parallel row chunks (fork
  context) and concatenated, giving byte-identical files.

The export of that cutout went from 131.9 s to 20.8 s. The `.dat` catalogues and the `.npz` are
byte-identical; only `code_sha256` in `physics.json` changes.

The worker then runs `skirt9_from_dat.py`, which re-parses those `%.17g` tables and rewrites them
at `%.10g` for SKIRT. Large tables are now parsed over line-aligned byte ranges and formatted in
row chunks, both in worker processes. On 7.8M stars this went from 19.4 s to 3.1 s, with identical
arrays and byte-identical output. Rendering (`finish_images`) takes about 2 s per galaxy.

## Large-mesh detector planes and batch failure handling

The first full-batch attempt stopped both shards on large environment cubes (9.1M and 16.4M
Voronoi cells, 10 cameras). The per-observer detector plane cache needs (4 N_cells + 20 N_faces)
bytes per camera, about 28 GB and 50 GB here, and an over-strict guard rejected more than 2³¹
faces × cameras. Detector rays now fall back to the shared transport planes when the cache would
exceed 40% of free GPU memory (`SKIRTGPU_DETECTOR_PLANE_BYTES` overrides the budget). The guard
now checks only cells × cameras. Forced shared mode gives identical images and is about 19% slower
on a small galaxy. The two failed galaxies completed in 89 s and 178 s (42 GB host peak).

The CFIS runner now records a failing galaxy in `galaxies/*/failure.json` and `failures.jsonl` and
continues with the next one. Disk shortfalls, pipeline or binary changes, an interrupt, or 10
consecutive failures still stop the shard. Recorded failures are skipped on resume unless
`run_batch.sh --retry-failed` is used.

## Validation summary

- `skirtgpu -G` passes on both GPUs after every change.
- Upstream CPU `skirt` references: production galaxy at 1e7 packets; viewer model with
  six self-absorption iterations (4,868 s on 32 threads, 97x the 49.9 s GPU run). Absorbed
  luminosities agree within 0.54% at every iteration (−0.34% at iteration 6). Integrated
  image flux differs by −0.29% to +0.34%, versus −0.95% to −1.25% for the deployed binary.
  Pixel L2 differences (≈30%) equal the GPU repeat-to-repeat Monte Carlo noise at 1e7 packets.
- Stochastic dust: `SKIRTGPU_VALIDATE_DUST_EMISSION=1` on every pass, ≤ 4.9e-14.
- Source luminosities: `SKIRTGPU_VALIDATE_SOURCE_LUMINOSITIES=1`, ≤ 1.7e-13.
- Remaining CPU/GPU per-band offsets on 99_640113 (u ≈ +0.4–0.6%, z ≈ −0.25 to −0.47%) exceed
  CPU seed-to-seed noise (≤ 0.28%) by about 0.3%. They also appear in full-FP64 GPU runs and in the
  old host path, so they predate this work and remain open.
