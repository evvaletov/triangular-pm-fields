# Correlated-fit verification

The numerical checks use synthetic observations and a specified fixed covariance.
No physical PM scan was available for this study.

Run from the repository root with the package installed, or set PYTHONPATH=python:

```sh
python validation/correlated/check_oracle.py --candidate python/triangular_pm_fields/correlated_fit.py
python validation/correlated/pm_gaussian_check.py --arrays validation/correlated/pm_fixture.npz --candidate python/triangular_pm_fields/correlated_fit.py --output local-results/correlated-gaussian.json
```

Create `local-results` first when running these commands manually.
`build_oracle.py` regenerates the six positive 80-digit small-matrix fixtures
and five rejection cases using mpmath. It uses tiny high-precision linear
solves, independently of the candidate SVD. The scalar-gain control checks the
parameter-variance floor and cancellation of shared gain in held-out residuals.
Dropping the train/holdout cross covariance produces the analytically specified
wrong residual covariance.

`PROTOCOL.txt` fixes the Gaussian experiment before the data were generated:
4,096 complete synthetic scans, a seed, covariance construction, and statistical
acceptance limits. The independent estimator uses whitened QR. Every candidate
scan is compared against its result. Shared gain, component offsets and global
observer-position sensitivities are included in the supplied covariance.

`pm_fixture.npz` contains 64 response locations and their coarse/fine geometry
sensitivities from the previous numerical PM-field study. The six parameters
are the lab-frame polarization components of two blocks. Sensitivities use
J=(0,1.2,0) T in each block. Global observer-displacement sensitivity is minus
the sum of the two source-center derivatives, by translation invariance.
These fixed arrays support a portable statistical check. They do not establish
new field accuracy or the validity of a nonlinear measurement-error model.

The fixture manifest records the source archive hash, selection, and individual
array hashes. `extract_pm_fixture.py` reproduces the selection when given that
original archive. The included fixture, seed and runner reproduce the
statistical experiment without the original archive. A real instrument model,
learned covariance and physical experimental agreement remain outside this check.
