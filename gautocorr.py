import numpy as np
import numba as nb


@nb.njit
def binary(t, v, i1, i2):
    m = i1
    while i1 < i2:
        m = (i1 + i2) // 2
        if t[m] < v:
            i1 = m + 1
        elif t[m] > v:
            i2 = m - 1
        else:
            break
    return m


@nb.njit
def gautocorr(tt, t, x, sigma=None, n_sigma=5):
    """
    Uses RBF kernel with distance `sigma`
    to evaluate the (positive) auto-correlation function
    for the function x(t) which can be sampled
    unevenly.

    Parameters
    ----------
    tt: Time lags to evaluate (positive numbers only)
    t: Function variable (assumed to be increasing)
    x: Function evaluated at `t`
    sigma: Standard deviation of the RBF
    n_sigma: Number of standard deviation to use in calculation

    Returns
    -------
    Auto-correlation

    """
    if sigma is None:
        sigma = np.mean(t[1:] - t[:-1])
    jsigma = n_sigma * sigma

    corr = np.zeros(len(tt))
    sw = np.zeros(len(tt))
    for ti, dt in enumerate(tt):
        for i in range(len(t)):
            a = t[i] + dt - jsigma
            if t[i] >= a:
                j0 = i
            else:
                j0 = binary(t, a, i, len(t))
            for j in range(j0, len(t)):
                tij = t[j] - t[i]
                dtij = tij - dt
                if dtij > jsigma:
                    break

                w = np.exp(-(dt - tij)**2 / (2 * sigma))
                corr[ti] += w * x[j] * x[i]
                sw[ti] += w
    corr = corr / sw
    return corr / corr[0]


# EXAMPLE USE BELOW
def scipy_autocorr(x):
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(corr)//2:]
    return corr / corr[0]


def example_fun(sim_n=500000, n=5000):
    t = np.empty(sim_n)
    x = np.empty(sim_n)
    t0 = 0.0
    x0 = 1.0
    v0 = 0.0
    for i in range(sim_n):
        t[i] = t0
        x[i] = x0

        dt = 0.015
        t0 += dt
        v0 += -0.25 * v0 * dt - x0 * dt + 0.2 * np.random.randn()
        x0 += v0 * dt + 0.2 * np.random.randn()
    x -= np.mean(x)

    ii = np.linspace(0, sim_n - 1, n)
    even_idxs = ii.astype(np.int64)
    f = lambda z: 1 - np.exp(3 * z)
    uneven_idxs = ((sim_n - 1) * (f(ii / sim_n) / f(1))).astype(np.int64)

    return t[even_idxs], x[even_idxs], t[uneven_idxs], x[uneven_idxs]


def example():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))

    t1, x1, t2, x2 = example_fun()
    assert len(t1) == len(x1) == len(t2) == len(x2)

    plt.subplot(2, 2, 1)
    plt.plot(t1, x1, '.-', lw=0.5, markersize=1.0)
    plt.title('Evenly sampled data')
    plt.subplot(2, 2, 3)
    # Scipy auto-correlation
    c = scipy_autocorr(x1)
    plt.plot(t1, c, label='Auto-correlation')

    # This auto-correlation
    tt = np.linspace(0, 100, 200)
    c = gautocorr(tt, t1, x1)

    plt.plot(tt, c, '--', label='Gaussian auto-correlation', alpha=0.75)
    plt.title('Nonuniformly sampled data')
    plt.legend()
    plt.xlim(-10, 100)

    # Now on unevenly sampled data
    plt.subplot(2, 2, 2)
    plt.plot(t2, x2, '.-', lw=0.5, markersize=1.0)

    plt.subplot(2, 2, 4)
    c = gautocorr(tt, t2, x2)
    plt.plot(tt, c, '--', c='tab:orange', label='Gaussian auto-correlation', alpha=0.75)
    plt.legend()
    plt.xlim(-10, 100)

    plt.show()


if __name__ == '__main__':
    example()
