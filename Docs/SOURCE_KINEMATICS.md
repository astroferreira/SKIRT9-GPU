# Resolved source velocities

The Observatory offers an opt-in **Source velocities** checkbox. The API field
is `source_velocities: true`; omission or `false` retains stationary sources.
This is a physical-model option, not an instrumental smoothing operation.
Existing jobs and source/dust catalogues are preserved. Velocity-on observations
and their derived input tables have separate cache identities.
The checkbox is disabled when the original cutout lacks velocity fields or
physical-export provenance. These requests are also rejected before queueing by
the API. Existing TNG cutouts downloaded without `Velocities` need a separate
data update; no velocities are synthesized and no particles are silently dropped.

## Physical scope

- BPASS emitters inherit their stellar particle's peculiar velocity.
- SFR-normalized TODDLERS emitters inherit their star-forming gas cell's velocity.
- Cosmological AREPO/TNG snapshot `Velocities` are converted to physical km/s by
  multiplying by `sqrt(a)`. The exporter validates its assumed velocity unit.
- The mass-weighted velocity of valid physical stars in the selected model cube
  is subtracted from both populations. Current stellar `Masses` are preferred;
  if absent, the explicitly recorded fallback is `GFM_InitialMass` weighting.
  The systemic frame can therefore change when the selected physical cube changes.
- Diffuse dust remains stationary. Scattering-induced shifts from moving grains
  are not included. Thermal/turbulent unresolved dispersion is not inferred from
  the effective temperature of star-forming simulation cells or added implicitly.
- RR2014 dust masses, Draine-Li grains, source luminosities/template parameters,
  source smoothing lengths, geometry, heating grids, CMB heating, and stochastic
  dust emission settings are unchanged.

Each source is Doppler shifted in its actual photon-launch direction. Its direct
contribution toward each observer uses that observer's direction independently.
The SKIRT nonrelativistic convention is
`lambda = lambda_rest * (1 - dot(direction, velocity)/c)`. Packet weight
`W = L * lambda` is preserved, including the observer-direction luminosity
adjustment. This preserves correlations between emission, source velocity and
the resolved dust column; a global convolution of the final SED would not.

TODDLERS already uses its high-resolution selected emission lines (R=50000).
SED detector resolving power is independent. R=3000 corresponds to roughly
100 km/s per resolution element; R=10000 corresponds to 30 km/s. Increasing
detector R does not recover missing template features or establish Monte Carlo
convergence. Dust-only RT is not a resonant gas-line transport calculation.

## Implementation

The exporter's `source_kinematics.py` derives separate source tables from saved
physical-export indices and the original HDF5 velocities:

```bash
.venv/bin/python source_kinematics.py BASE.ski MODEL_physics.json NEW_DIRECTORY
skirtgpu -g -i NEW_DIRECTORY -o NEW_DIRECTORY NEW_DIRECTORY/model.ski
```

It rejects changed raw-file identities, unsupported schemas, missing velocity
fields, invalid particle selections and mismatched source positions/counts.
It preserves all nonvelocity source columns, leaves dust inputs untouched, and
records input/output hashes, units, systemic velocity and population assignment.
Files are never overwritten. The viewer runs this in the exporter environment
and publishes the completed cache directory atomically.

GPU changes are in `PhotonPacket`, `ImportedSource`, `MonteCarloSimulation`,
`MediumSystem` and `GpuAcceleration`:

- Packets capture their launch velocity by value. The previous thread-local
  interface could be overwritten before a batched packet's detector contribution.
- Host spectral sampling and GPU direction generation remain batched. No full
  per-particle BPASS spectral cache is introduced.
- The resident initial-emission detector accepts an optional four-double payload
  per active packet: rest wavelength and three velocity components. It evaluates
  each observer's Doppler shift, luminosity and extinction-table lookup on GPU.
- Subsequent propagation/scattering uses the existing resident GPU transport for
  stationary dust. The payload is not allocated for velocity-off batches.
- The existing two-device scheduling and radiation-field accumulation remain.

The accelerated moving-source path currently requires the default mixed-precision
resident Voronoi transport with one supported HG dust mixture and distant
Frame/SED detectors. Disabling that path or enabling unsupported media/detectors
retains the scalar/batched fallback; it does not silently discard velocities.
The direct device source-spectrum cache path still rejects moving sources; the
production viewer already disables that cache to bound memory.

Raw velocity storage costs 24 bytes per emitter in double precision, plus
packet staging/device buffers. Actual peak memory and runtime must be measured
for the chosen model, photon count, wavelength grid and instruments.

## Validation

`tools/validate_source_velocities.py` runs actual pristine CPU SKIRT and this GPU
binary on Gaussian line sources in a Voronoi dust mesh. Tests cover zero velocity,
a common bulk shift, unequal-luminosity counter-streaming sources, an imported
isotropic dispersion, and a non-negligible dust column, each at three viewing
directions. All five cases passed analytic and CPU/GPU centroid, width and flux
checks. Every GPU case confirmed resident transport. These small tests are not
production timing measurements or a general CPU/GPU parity guarantee.

Exporter tests cover sqrt(a) conversion, current-mass systemic weighting,
wind/future-star exclusion, young-SSP fallback, exact unchanged source columns,
raw identity and position validation. Viewer tests cover opt-in behavior,
cache separation/reuse, table offsets, preview preservation and missing provenance.
The browser test verifies the checkbox, request payload, saved-state restoration,
and desktop/mobile layout without launching a simulation.

Matched production timings and physical FITS/SED products are recorded under
`/home/astroferrreira/SKIRT/diagnostics/source_velocities_20260911/`.
Consult `benchmark.json` for measured completed runs; a running entry is not a
performance result. The initial relocated-binary resource lookup failure occurred
before simulation setup and is excluded from timings. An incomplete numbered
source-table header was caught and corrected with a real-SKIRT parser regression.
A trial interrupted by a new viewer observation is also excluded. The benchmark
stops its own process whenever it detects an active viewer job.

### Completed live observation and timing limit

The actual on-demand MOOSE-197 50 kpc velocity-enabled SED observation
`visualizer/var/jobs/1b8955a486d342129befbf943b961e6a/` completed successfully:
10M packets per phase, MIRI coverage with ideal R=1000, 153 s SKIRT runtime,
32.2 s primary emission, 8.96 GB reported peak host memory, and 163.9 s including
viewer preparation and product processing. Both RTX 5090 devices were enabled;
3481/3481 spectral bins were finite and positive. The manifest records unchanged
RR2014/DL07 diffuse dust mass 209214895.0305935 Msun and the velocity provenance.

The matched old-binary RGB+SED control completed in 155 s, including 32.0 s
primary emission. A velocity-on trial completed primary emission in 32.3 s before
being interrupted during dust emission by a viewer job. Its unfinished full run
is not a timing result. The live SED-only run above has different instruments and
viewing direction: **it is not a matched speedup measurement**. At the user's
request, repeated matched timings were stopped so viewer GPUs remain available.
No 1000x speedup, full CPU parity, or unchanged production runtime is claimed.

The live model retains the BPASS long-wavelength coverage warning. Enabling
source velocities also triggers a Draine-Li short-wavelength coverage warning:
SKIRT's generic v/c=1/3 safety margin requests 0.0371 micron while these grain
tables begin at 0.04 micron. The measured maximum source speeds in this model
are 853.9 km/s (stars) and 403.8 km/s (TODDLERS), so 0.05 micron source photons
remain above 0.04985 micron after launch shifts. Stationary grains do not shift
them further; the unchanged heating/cooling grids lie within coverage. The
warning is preserved, not suppressed or addressed by altering the dust mass.
This coverage argument must be reconsidered for different wavelength limits or
very fast sources. No grain-property extrapolation was added.

The existing nearly-zero cosmological redshift of MOOSE-197 implies an
unphysical near-zero luminosity distance for this live SED. Its absolute flux
normalization must not be interpreted as a finite-distance galaxy observation;
the separate benchmark uses an explicit local 10 Mpc observer. This pre-existing
distance setting was not changed as part of the velocity implementation.

## References

- [SKIRT snapshot velocity import](https://skirt.ugent.be/root/_user_import_snap.html).
- [TNG snapshot and group-catalogue velocity conventions](https://www.tng-project.org/data/docs/specifications/).
- [TODDLERS spectral resolution and line templates](https://skirt.ugent.be/skirt9/class_toddlers_s_e_d_family.html).
