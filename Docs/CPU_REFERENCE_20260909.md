# Unmodified upstream CPU baseline: 2026-09-09

The CPU reference is now a separate, pristine upstream SKIRT9 worktree, not
the GPU fork with GPU flags omitted. Its pinned revision is
`dcac71e5cadfb66329633137ccdd616cb404c94a`, the parent of the first GPU commit.
It was verified to be an ancestor of the official upstream master after a
fresh fetch. Newer upstream changes were deliberately excluded so the
comparison uses the original base of the GPU port.

```
Source: /home/astroferrreira/SKIRT/cpu-reference/git
Binary: /home/astroferrreira/SKIRT/cpu-reference/release/SKIRT/main/skirt
Notes:  /home/astroferrreira/SKIRT/cpu-reference/README.md
```

The build is GNU C++ 15.2.1, Release, `-O3 -DNDEBUG`, C++14. The source
worktree is clean and contains no `GpuAcceleration`, `SKIRTGPU`, or CUDA
launch implementation. Built-in resource files match the GPU checkout;
external resource packs are shared through a symlink. The running CPU
process had no CUDA/NVRTC libraries mapped and no NVIDIA compute context.

## Matched observation

TNG50-1 snapshot 50 / subhalo 257247, z=1, halo-mode 150 kpc export, Nk16,
four 1000 x 1000 views, eight NIRCam and nine MIRI broadband filters, and four
integrated spectra sampled over 0.7-5.3 micron for ideal R=1000 rendering.
There are 10 million primary and 10 million secondary packets, seed 67890,
with stochastic dust emission. Both runs use 32 host threads on an AMD
Ryzen Threadripper 9960X (24 physical cores, 48 logical CPUs). This does not
establish the optimal CPU thread count.

Both executables were supplied exactly the same SKI file:

```
/home/astroferrreira/SKIRT/benchmarks/z1_jwst_20260905/z1_jwst_1k_10m_seed67890.ski
```

The fully expanded model XML, including seed and packet count, is identical.
Canonical SHA-256:
`ecda9d621569117996c344820ccd4dc77e88aa1af1b1c76eebcc4bbcc8f131ae`.

## Completed timings

| Stage | Upstream CPU | Two RTX 5090 GPUs | CPU / GPU |
| --- | ---: | ---: | ---: |
| Setup | 34.0 s | 33.2 s | 1.02x |
| Primary emission | 316 s | 96 s | 3.29x |
| Secondary emission, including dust spectra | 574 s | 51.3 s | 11.19x |
| Run phases, SKIRT-reported | 891 s | 147 s | 6.06x |
| Entire simulation | 925 s | 181 s | 5.11x |
| Peak host memory | 5.61 GB | 15.5 GB | - |

Stage timings are rounded independently by SKIRT. The CPU ran alone, after
the GPU test and reference build had finished. Build time is not included.
The 181 s GPU result is the existing dense-table observation, not the
experimental compact-layout result. The prior 329x photon-shooting / 168x
end-to-end SDSS benchmark is a different galaxy and workload and must not be
applied to this JWST observation. The active 500x target is not achieved.

Evidence under `/home/astroferrreira/SKIRT/benchmarks/z1_jwst_20260905/`:

```
run_cpu_upstream_1k_10m_seed67890/
run_1k_10m_gpu_stochastic_seed67890/
cpu_upstream_jwst_comparison.json
```

## Output checks and limitations

All four CPU FITS cubes have shape (17, 1000, 1000), finite nonnegative
pixels, and positive flux in every band. Wavelength axes and units match
the GPU outputs. All four 4051-sample SEDs are finite and nonnegative, with
identical wavelength axes. Grid convergence reports are byte-identical.

**This is a timing measurement, not a verified CPU/GPU parity result.**
Band-integrated differences reach 5.60% (xz, MIRI F1130W); the corresponding
F1280W difference is 5.48%. Integrated 0.7-5.3 micron spectral differences
are 1.24-1.87%, with GPU flux lower in all four views. These differences
must be investigated with additional realizations and estimator checks;
they must not simply be attributed to Monte Carlo noise. The scalar
GPU dust-emissivity checks at fixed radiation fields do not validate the
entire transport and detector workflow.

Equal photon counts do not establish equal noise: the GPU can sample
observers while the CPU records all observers. The existing multithreaded
CPU source generator also intentionally uses operating-system-seeded
worker generators (`Random.hpp`), so a fixed configured seed does not
guarantee pixel-identical hybrid or CPU runs. Single-run pixel differences
are not in themselves a deterministic regression test.

Both observations retain the dust-optics coverage warning: the configured
required range extends to about 2020 micron, but the dust tables end at
1000 micron. The first-pass images and spectra are not noise-converged,
PSF-convolved, detector-noise-added JWST observations.

The earlier interrupted `run_cpu_1k_10m_seed67890` used the modified fork
and created a CUDA context through cache invalidation despite reporting
GPU-disabled status. It is preserved but excluded from this benchmark.
The fork's invalidation path was changed to avoid initializing an unused
runtime pool. The reference above does not depend on that fix.
