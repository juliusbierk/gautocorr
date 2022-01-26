# Gaussian auto-correlation for unevenly sampled data

This is a naive implementation using RBF kernels of the auto-correlation function
for a function that is nonuniformly sampled.
Usage

```python
import numpy as np
from gautocorr import gautocorr
t = 10 * np.linspace(0, 1, 100)**2  # uneven time points
x = np.cos(t)  # function
tt = np.linspace(0, 3, 20)  # time lags to evaluate auto-correlation at
correlation = gautocorr(tt, t, x)
```

For more involved example see `gautocorr.py`.

Requires `numpy` and `numba`.
