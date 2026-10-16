# SED reference features

The integrated SED viewer overlays reference wavelengths for common nebular
lines, rotational molecular-hydrogen lines, PAH complexes and silicate features.
Category checkboxes affect the overlay only. Selecting a plot label, a nearby
vertical marker, or an entry in the reference menu displays its physical meaning
and a source link. The menu retains features whose labels cannot fit in a crowded
plot. Zooming recomputes visible references; responsive label lanes avoid overlap.

These are identification aids, **not detections, fits, flux measurements or a
claim that every feature is represented in the selected source templates**.
High-ionization references do not add an AGN to a BPASS/TODDLERS model. PAH and
silicate markers are nominal complex positions, not narrow emission lines or
predictions of the precise band peak. Their shape and emission/absorption balance
depend on the grain population and radiative transfer.

## Frames and preservation

Observer-frame markers use `lambda_observed = lambda_rest * (1 + z)` with the
redshift in that job's saved `observation.ski`. A fixed-distance rest-frame
observation uses zero shift, even if its simulation snapshot is cosmological.
The current catalogue selection never supplies an old observation's redshift.
Missing/unsupported redshift metadata disables observer-frame references rather
than guessing. Bulk source velocities can shift individual peaks relative to the
systemic markers; these labels do not fit or undo that broadening.

`reference_redshift` is added to new spectral JSON metadata. The spectrum API
enriches older responses in memory, without rewriting saved JSON, DAT, FITS, CSV
or PNG products. The overlay is browser-side and requires no new Monte Carlo run,
particle download or GPU work. It is not baked into the downloadable science data
or existing static preview PNGs.

## Sources and precision

The curated catalogue is `visualizer/frontend/spectral-features.js`.
Optical values use the SDSS vacuum reference convention. Near-IR hydrogen values
and broad-band centres are rounded identification references. These are not a
homogeneous precision wavelength-calibration database; do not infer small radial
velocities from the label positions.

- [SDSS vacuum spectral references](https://classic.sdss.org/dr7/algorithms/linestable.php).
- [ISO atomic-line reference compilation](https://www.ipac.caltech.edu/iso/sws/atomic.html).
- [ISO molecular hydrogen reference compilation](https://www.ipac.caltech.edu/iso/sws/HII.html).
- [PAHFIT classic molecular-hydrogen reference values](https://raw.githubusercontent.com/PAHFIT/pahfit/master/pahfit/packs/science/classic.yaml).
- [Hydrogen recombination diagnostics](https://ned.ipac.caltech.edu/level5/March01/Dinerstein2/Diner2.html).
- [Smith et al., PAH emission and silicate extinction](https://arxiv.org/abs/astro-ph/0610913).
- [Farrah et al., infrared galaxy line diagnostics](https://arxiv.org/abs/0706.0513).

Tests cover saved observer/rest frames, unknown redshifts, preserved original
products, group and zoom filtering, unique references, label collisions,
clickable meanings, and desktop/mobile rendering. Browser tests use mocked
spectra and existing real observations, never submit a new GPU render by default.
