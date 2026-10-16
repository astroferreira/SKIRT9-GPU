# Optical broadband performance: 2026-09-27

The current local `skirtgpu` produces four 600 x 600 SDSS ugriz images in
3.69 s and 2.75 s for two matched TNG50 models, versus 64.92 s and 106.95 s
with the pristine upstream CPU executable. These are **17.6x and 39.0x
end-to-end speedups with a warm compiled-kernel cache**, including process
startup and final FITS output. They are 2.02x and 2.35x faster than the
preceding GPU executable on these same broadband workloads.

The first uncached candidate run took 7.61 s for 706717. Cold startup did
not improve; most of the repeated-job gain comes from avoiding repeated
NVRTC compilation. The numerical transport changes separately reduced
secondary-emission time by 31% and 46%.

## Changes

- Mixed-precision resident emission and scattering detectors test spectral
  response before tracing observer extinction. Rejected rays have exactly
  zero contribution under the existing band interpolation and SED bin rules.
  Frame and SED redshifts are checked separately, and emission Doppler shifts
  are applied before the test.
- After emission detection, the elastic HG lifecycle can terminate packets
  whose transport wavelength cannot contribute to **any** observer. This
  is allowed only when no radiation field is being stored. Primary photons
  still heat dust outside the observed filters, and direct emission from
  moving sources is preserved. `SKIRTGPU_BROADBAND_CULL=0` disables this
  transport cull for diagnostics; detector rejection remains enabled.
- NVRTC PTX is cached across processes. Entries contain the complete CUDA
  source, compiler major/minor version, exact compiler options, and a payload
  checksum. Writes are atomic. Missing, invalid, or unwritable entries fall
  back to compilation; a driver-rejected module is recompiled once while
  bypassing both caches. No simulation data is stored in this cache.
- `tools/compare_broadband_runs.py` verifies expanded-model equality,
  expected Frame outputs, shapes, units, wavelength axes, finite nonnegative
  pixels, per-band fluxes, and stage timings. It supports built-in bands and
  file bands, including inherited instrument grids.

Persistent compilation caching is automatic. The default directory is
`$XDG_CACHE_HOME/skirtgpu/nvrtc`, or `$HOME/.cache/skirtgpu/nvrtc` if XDG is
unset. `SKIRTGPU_KERNEL_CACHE_DIR` overrides the directory,
`SKIRTGPU_KERNEL_CACHE=0` disables persistent caching, and
`SKIRTGPU_PROFILE_KERNEL_CACHE=1` reports hits and compilation. The normal
in-process cache is retained when persistent caching is disabled.

## Matched runs

Both galaxies are TNG50-1 snapshot 99, using the existing halo-mode BC03,
MAPPINGS, Voronoi dust, and equilibrium dust-emission configurations.
Benchmark copies remove SED instruments and set the packet counts below;
source wavelengths, dust physics, seed 12345, spatial resolution, and
viewing directions are otherwise retained. CPU and GPU receive the exact
same SKI for each case. No photon counts, filters, or dust-emission phases
are reduced on the GPU.

| Galaxy | Packets per primary/secondary phase | CPU wall time | Previous GPU median | New GPU median | CPU/new GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 706717 | 16,777,216 | 64.917 s | 7.427 s | 3.685 s | 17.62x |
| 746970 | 33,554,432 | 106.950 s | 6.461 s | 2.745 s | 38.95x |

Times are external process wall times, not rounded SKIRT log totals.
Each GPU median has three completed sequential runs; each CPU time is one
completed run. GPU ranges are 3.677-3.687 s and 2.744-2.797 s. CPU and GPU
timings did not overlap, and builds did not overlap CPU timings.

Hardware: two RTX 5090s and an AMD Ryzen Threadripper 9960X. Both CPU and GPU
use 32 host threads; this is not an optimal-CPU-thread-count claim.
The GPU runs explicitly set `SKIRTGPU_RESIDENT_SAMPLE_OBSERVER=0`, so all
four observers are evaluated, without the high-count observer-sampling
variance tradeoff. The user-authorized stop of `qwen38-api.service` freed
GPU 1 before all timed runs; that service remains stopped.

The reference is the clean upstream checkout at
`dcac71e5cadfb66329633137ccdd616cb404c94a`, described in
[CPU_REFERENCE_20260909.md](CPU_REFERENCE_20260909.md).

Phase medians measured from millisecond log timestamps:

| Galaxy | Previous primary | New primary | Previous secondary | New secondary |
| --- | ---: | ---: | ---: | ---: |
| 706717 | 1.747 s | 1.715 s | 0.348 s | 0.239 s |
| 746970 | 1.077 s | 1.073 s | 0.333 s | 0.180 s |

## Validation and limits

All 40 SDSS image bands have matching dimensions/units/wavelengths and
finite nonnegative pixels. Maximum integrated CPU/GPU band differences
are 0.313% and 0.488%. This does **not** establish equal image noise or
complete CPU/GPU physical parity: CPU/GPU per-band relative pixel L2
differences reach 16.5% and 5.79%, respectively. The existing GPU analog
estimator and CPU forced-scattering estimator differ.

The new versus previous GPU results differ in integrated band flux by at
most 1.25e-9 relative. Peak-normalized pixel errors are at most 2.77e-5
and fall within the observed unchanged-executable repeat envelope. An
initial absolute 1e-5 peak-error gate failed, but the original executable
also differs from its own repeat by 2.72e-5. No bitwise-repeatability claim
is made. The existing model dust-wavelength-coverage warning is unchanged.

Additional checks passed:

- Both-GPU `skirtgpu -G`, including 60 spectral support cases covering band
  endpoints, zero-transmission intervals, SED gaps, merged Frame/SED
  acceptance, separate redshifts, and accumulator bounds.
- Both-GPU self-tests with persistent caching disabled and with a deliberately
  invalid but checksummed PTX entry, exercising driver-rejection recovery.
- Five one-million-packet moving-source CPU/GPU analytic cases: stationary,
  bulk motion, split velocity, velocity dispersion, and dusty scattering.
  The test template's wavelength bias was adjusted to cover its Ly-alpha line.
- CFHT u/g/r file-band images with four simultaneous SEDs: image regression
  peak errors below 3.71e-5 and spectrum relative L2 errors below 6.49e-10.
  These use the existing approximate MegaCam response files; they are
  compatibility checks, not new calibrated CFHT throughput curves.
- Seven analyzer tests; standalone C++14 cache tests, including corruption,
  truncation, changed keys, disabled/unusable paths, and concurrent writers.
- Release builds of `skirtgpu` and `skirt`; `git diff --check`.

## Evidence and reproduction

All inputs, runs, external timings, profiles, JSON comparisons, retained
baseline executable/source, and RGB preview are under:

```
/home/astroferrreira/SKIRT/benchmarks/optical_broadband_20260927/
```

`summary.json` holds measured speedups and image errors. `run.py` creates
new numbered output directories and refuses to overwrite existing runs.
`summarize.py` regenerates the numerical report. The `after_1` directories
contain the generated FITS images, and `sdss_gri_preview.png` displays all
eight views with a common-per-image RGB display stretch.

Example for a fresh output directory:

```bash
mkdir -p /tmp/skirtgpu-optical-example
SKIRTGPU_MAX_DEVICES=2 SKIRTGPU_RESIDENT_SAMPLE_OBSERVER=0 \
  /home/astroferrreira/SKIRT/release/SKIRT/main/skirtgpu -t 32 \
  -o /tmp/skirtgpu-optical-example \
  /home/astroferrreira/SKIRT/benchmarks/optical_broadband_20260927/706717/optical.ski
```

Final GPU binary SHA-256:
`20863cb4ace941ec887fbf1c4c7611add7dd271013d676dcc1e4c612f2c0ad41`.
Baseline GPU:
`393b6badffbfd1d499c3b977c90bc7fc868fa5401f07cb5076c59efd4f0edd9c`.
Upstream CPU:
`2464ebb100833ff3b7360056afd437e72994f5da0930e17de486e962ebe0ddac`.

The next measured bottleneck is primary source/lifecycle preparation and
remaining setup work. Broadband culling alone barely changes primary time
on these cases because the primary radiation field must still be computed.
