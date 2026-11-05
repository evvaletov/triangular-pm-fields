# Python analytic baseline

`triangular_pm_fields.reference` contains the original NumPy/complex-jet
closed form and its surface-quadrature helpers. It is a readable standalone
baseline. The production Python native client is `NativeLibrary`.

```python
import math
import numpy as np
from triangular_pm_fields.reference import field, field_jet, indices

vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
point = np.array([.25, .25, .75])
sigma, radius, order = 1.25, .125, 6
mu0_h = sigma / (4*math.pi) * field(vertices, point)
jets = field_jet(vertices, point, order)
coefficients = np.array([
    [jet.c.get(alpha, 0j).real * radius**sum(alpha) * sigma/(4*math.pi)
     for alpha in indices(order)] for jet in jets
])
```

`field` and `field_jet` return the unscaled kernel `K`. The latter returns
three `Jet` objects with coefficient dictionaries. Their increments have the
same physical length unit as the input coordinates. Apply the charge and
radius factors explicitly as above.

The native interfaces and new standalone ports have stricter input/status
contracts. This baseline uses floating geometry and is qualified here on
the 48 regular-center Taylor cases. It does not supply the exact binary64
boundary predicates or status protocol of those interfaces. The independent
release oracle is implemented separately in `tools/generate_reference.py`;
its quadrature is not a call to this candidate baseline.
