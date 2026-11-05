# Linear fitting with a known joint covariance

`CorrelatedLinearFit` fits a fixed linear response to training observations
and predicts a separate set of held-out observations. It accepts correlations
within and between scans. The caller supplies the response matrices and the
complete, known covariance before fitting.

```python
from triangular_pm_fields import CorrelatedLinearFit

model = CorrelatedLinearFit(A_train, A_holdout, C_joint,
                            parameter_scales=scales)
fit = model.fit(y_train)
validation = model.evaluate_holdout(y_train, y_holdout)
```

`A_train` has shape `(n_train, n_parameters)` and `A_holdout` has shape
`(n_holdout, n_parameters)`. Observations are one-dimensional arrays. A vector
field can use consecutive x, y, z rows for each observer. A scalar probe needs
rows for its measured projection. The class does not infer component order,
coordinate transforms, units, or calibration from the arrays.

`C_joint` has shape `(n_train+n_holdout, n_train+n_holdout)` with all training
rows first. Its diagonal contains variances, not standard deviations. It must
be finite, real, symmetric to roundoff, and positive definite. The class rejects
singular covariance matrices and adds no diagonal jitter. Parameter scales are
positive reference sizes used to scale columns for rank assessment; they impose
no prior or regularization. An omitted scale defaults to one for each parameter.
The class also rejects nonfinite arithmetic and a parameter or residual variance
that underflows to zero. Rescale the physical units if the problem exceeds the
supported float64 range.
Parameter scales, nonzero column-scaled design entries and nonzero singular
values must stay in the normal float64 range. Destructive subnormal rounding
or a nonzero design entry that underflows during scaling causes a range error.

The fixed model is

```text
y_train   = A_train theta + e_train
y_holdout = A_holdout theta + e_holdout
Cov([e_train, e_holdout]) = C_joint
```

Cholesky whitening followed by an SVD gives the estimator
`theta_hat = H y_train`. Designs with fewer training rows than parameters, or
with deficient numerical rank, are rejected. The rank cutoff is
`max(n_train,n_parameters)*eps*largest_singular_value` after whitening and
column scaling. The normal matrix is not inverted.

Prepared arrays and fit results are immutable copies. The prepared object
exposes `estimator`, `parameter_covariance`, `singular_values`, `rank`,
`condition_number`, and `training_dof`. `fit()` returns parameter estimates,
training predictions and residuals, held-out predictions, training chi-square,
and training degrees of freedom. Fitting uses only the training observations.

Let `K = A_holdout H`. The held-out residual is
`r = y_holdout - K y_train`. For covariance blocks ordered as above,

```text
Cov(r) = C_hh + K C_tt K.T - C_ht K.T - K C_th.
```

The implementation computes this covariance from the joint Cholesky factor:
`F = [-K, I] L`, `C_joint = L L.T`, `Cov(r) = F F.T`. This avoids subtracting
large covariance terms. `holdout_residual_covariance` includes error in both
scans. `prediction_covariance = A_holdout Cov(theta_hat) A_holdout.T` describes
uncertainty in the fitted mean prediction and excludes held-out measurement
noise. These two quantities answer different questions.

`evaluate_holdout()` returns the fit and the held-out residual and chi-square.
For zero-mean Gaussian errors with the specified fixed covariance and a correct
full-rank linear model, the training statistic has `n_train-n_parameters`
degrees of freedom, and the held-out statistic has `n_holdout`. Both statements
average over repeated training and held-out observations. The predictor is
`A_holdout theta_hat`; it does not condition the held-out noise on the observed
training residuals. Non-Gaussian errors retain the covariance propagation but
do not generally give these chi-square distributions.

## Calibration and physical interpretation

A shared gain error can leave a small held-out residual while shifting the
estimated magnet polarization. The gain contribution remains in parameter
uncertainty even when it cancels between scans. Agreement between scans alone
therefore cannot establish the physical field accuracy.

Gain, probe alignment, position and geometry errors can contribute through a
fixed linearized covariance. Their sensitivity coefficients must come from
independent specifications or a predeclared nominal model, with the relevant
linearization checked. The software does not estimate these coefficients from
the fitted residuals. A covariance that depends on unknown polarization or
geometry needs a separately qualified inference procedure. Free probe gain and
free magnet polarization are not jointly identifiable without an external
constraint.

No physical PM scan was available for this release. The statistical evidence
uses synthetic fixed-covariance models and does not qualify a real instrument,
nonlinear calibration model, learned covariance, or experimental agreement.
The dense joint covariance uses quadratic storage and a cubic-cost Cholesky
factorization, so this interface is intended for modest scan sizes.
