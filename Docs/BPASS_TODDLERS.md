# BPASS/TODDLERS physical model and stochastic dust launch

The exporter under `~/src/illustris_to_stellarmap` now supports explicit
BPASS/TODDLERS catalogues. See its `PHYSICAL_MODEL.md` for schemas, age/SFR
accounting, the empirical diffuse/birth-cloud prescription, and usage.
The four 2026-09-10 versioned exports and measured validation products are
under `~/SKIRT/diagnostics/skirt_physical_upgrade_20260910`. Original
catalogues and pre-existing source edits are preserved.

Required resource packs are BPASS 3, TODDLERS 2, ExtraDust 3, Core 8 and
ExtraBands 1. Only SFRNormalized TODDLERS is used; the large Cloud-mode
resource packs are unnecessary. Resource version numbers do not identify
a scientific-paper release.

The default spectrum is BPASS Chabrier100/downsampled for old SSPs, plus
TODDLERS BPASSChab100Bin/Period10Myr/High with explicit SFE and density.
The diffuse medium uses DraineLiDustMix and stochastic heating. The
metadata reports template bounds and does not claim a measured combined
diffuse+implicit-cloud dust mass.

`DustSecondarySource` reuses its prepared GPU batch spectra during host
photon launch. This prevents secondary self-absorption iterations without
peel-off from recalculating stochastic heating for every emitting cell.
Host PDF/CDF normalization remains in double precision; the existing
prepared PDF storage is float. Cache identity includes the simulation/source
token, iteration generation, and launch-order cell, including compact-row
lookup. Scalar fallback invalidates the prepared cache.

Set `SKIRTGPU_HOST_PREPARED_DUST=0` to retain the original host calculation
for comparisons. `SKIRTGPU_VALIDATE_DUST_EMISSION=1` checks prepared
emissivities against the CPU calculation. The validation directory records
on/off runs, independent CPU seeds, resource axes, and the kernel self-test.
This correction does not establish the separate 1000x production speedup
criterion or fine-pixel convergence.

## GPU utilization follow-up (2026-09-11)

The BPASS and SFR-normalized TODDLERS families now implement `cdfBatch`,
using the existing GPU stored-table integration kernel and preserving the
scalar mass, age, density, and SFR unit conversions. Large batches are
distributed over all available GPUs. Cloud-mode TODDLERS still uses the
scalar fallback. `SKIRTGPU_SOURCE_LUMINOSITIES=0` disables this setup path;
`SKIRTGPU_VALIDATE_SOURCE_LUMINOSITIES=1` checks up to 256 catalogue entities
per source family against scalar integration and stops on a mismatch.

The viewer keeps full per-particle spectral caches disabled: the BPASS
8000-point grid makes those caches too large for million-particle models.
Primary wavelength sampling is therefore still on the CPU. The viewer now
uses its configured host thread count for this work rather than the legacy
six-feeder limit. This is not a GPU-only source implementation.

(Superseded on 2026-09-30: the warp-per-cell dust kernel in
[THROUGHPUT_20260930.md](THROUGHPUT_20260930.md) needs no scratch matrix; the
default batch is now 16384 cells and the limit 1048576.) Dust emission batch sizes
could be selected with `SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS=1..1024` (default 256); the viewer
exposes this through `SKIRT_VIEWER_DUST_BATCH_CELLS`. Larger batches improve
GPU occupancy at the cost of temporary matrix memory. The numerical
algorithm, temperature grids, self-absorption, and source catalogues are
unchanged. Do not disable dust self-absorption to reproduce these timings.

On the unchanged MOOSE-197 50 kpc, 100M-packet model, the original viewer
run took 284 s. The candidate with 1024-cell dust batches and 16 primary
feeders took 142 s: 2.00x faster than the previous GPU version, not CPU
SKIRT. Primary emission improved from 91 to 36 s, setup from 37.2 to 13 s.
Source checks agreed within 7.3e-14 relative error; sampled dust emissivity
checks agreed within 3.3e-15 with no CPU fallback. Band-integrated image
flux differed by 0.064%; this is not a convergence or full CPU-parity test.
Configured seeds do not fix the OS-seeded hybrid worker RNGs.

The expanded observation XML was identical. Evidence and further tests:
`/home/astroferrreira/SKIRT/diagnostics/gpu_utilization_20260911/`.
The interrupted profiled baseline there overlapped a user render and must
not be used as a timing reference. The original completed viewer job
`f26561f44aee4a5f91e1d1b419095551` supplies the 284 s reference.

The lower-memory 512-cell / 16-feeder variant completed in 158 s (1.80x
versus the original GPU version), with primary emission 35.6 s and
integrated R-band flux difference -0.0209%. This is the selected local
dual-5090 viewer configuration. The CLI dust batch default remains 256.

## Corrected resolved dust (2026-09-11)

The user requested the previous resolved-dust prescription while retaining
BPASS/TODDLERS and corrected source accounting. The selected viewer default
combines RR2014 dust masses with DraineLiDustMix (15 size bins per population),
without erasing whole star-forming cells. This is a physical-model
change, not a GPU acceleration setting. It remains an empirical diffuse plus
implicit-cloud model, not a demonstrated conserved combined dust budget.

The exporter accepts `--diffuse-dust rr2014 --diffuse-grain-mix dl07`; its old
CLI defaults remain Torrey/DL07 for compatibility. The separate comparison uses
`--diffuse-grain-mix wd01`, with five sizes per population. WD01 optical properties end at 1000 micron,
so this alternative limits source and rest-frame grid centres to 900 micron
including a bin-edge safety margin. Its longer-wavelength TODDLERS tail is
omitted. Matched tests apply the same limit to every variant.

`derive_diffuse_dust.py` in the exporter derives new dust tables from saved gas
fields while checking original hashes and keeping exact mesh coordinates and
unchanged source files. The on-demand viewer export cache includes explicit
mass-prescription and grain-mixture choices, preventing reuse of an incompatible
cached region. Original catalogues, renders, and legacy models are preserved.

Staging, deployment status, comparison FITS, column/optical-depth probes and
scientific limitations: `/home/astroferrreira/SKIRT/diagnostics/resolved_dust_20260911/`.

The completed WD01 control exposed an inherited cooling-range warning: SKIRT
always integrates the Planck cooling functions through 2000 micron, regardless
of the transported spectral range. The 900 micron limit does not fix that
internal coverage gap. RR2014/DL07 is therefore the selected production
choice, retaining the original 0.05-2000 micron source range and grids; WD01
remains a separate comparison. No opacity-table extrapolation or C++ cooling
change was introduced to suppress the warning.

The eight revised catalogue entries and on-demand raw-series exports select
RR2014/DL07 explicitly. Prior Torrey/DL07 entries, completed observations and
WD01 diagnostic controls are preserved; full-halo rendering remains disabled.

A fresh live on-demand MOOSE-197 run verified RR2014/DL07, unchanged source
families, 2.09215e8 Msun imported diffuse dust and both RTX 5090 devices. The
10M-packet, 512-square i/r/u observation completed in 151 s in SKIRT (167.48 s
including viewer export and image processing). It retained the 0.05-2000 micron
range and reached self-absorption convergence in two iterations. The FITS are
finite and nonblank, but native pixels remain noisy; no image convergence or
CPU/GPU speedup follows from this test. Job and provenance:
`visualizer/var/jobs/778a20cbb00e4643b276cd34d1318454/manifest.json`.

## Optional resolved velocities

[Source kinematics](SOURCE_KINEMATICS.md) documents the opt-in stellar and
star-forming-gas velocity import, resident GPU Doppler detector, validation and
matched runtime tests. This leaves the RR2014/DL07 medium stationary and does
not change its mass or heating model. Existing observations remain unchanged.
