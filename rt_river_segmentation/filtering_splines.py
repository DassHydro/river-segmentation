import numpy as np
from scipy import interpolate, signal, stats

import matplotlib.pyplot as plt

def hydraulic_filtering_spline(x, h, x_direction="downstream", plot_steps=False):
    """ Compute regression spline that preserves hydraulic characteristics, function call similar to rt_river_segementation
    """

    dx = x[1] - x[0]
    if x_direction != "downstream":
        raise NotImplementedError("Upstream x-direction not implemented yet")
    return hydraulic_preserving_spline(x, h, plot=plot_steps)


def hydraulic_preserving_spline(x, h, dx=10, seps=1e-9, le_min=20000.0, plot=False):
    """ Compute regression spline that preserves hydraulic characteristics
    """

    N = x.size

    # Compute number of extrapolation points
    Ne = N//4
    le = Ne * dx
    if le < le_min:
        Ne = int(np.ceil(le_min / dx))

    # Extrapolate near boundaries using linear regression
    res = stats.linregress(x, h)
    xs = np.linspace(x[0]-Ne*dx, x[0], Ne+1, endpoint=True)
    hs = xs * res.slope + res.intercept
    hs += h[0] - hs[-1]
    xe = np.concatenate((xs[:-1], x))
    he = np.concatenate((hs[:-1], h))
    xs = np.linspace(x[-1], x[-1]+Ne*dx, Ne+1, endpoint=True)
    hs = xs * res.slope + res.intercept
    hs += h[-1] - hs[0]
    xe = np.concatenate((xe, xs[1:]))
    he = np.concatenate((he, hs[1:]))

    # Compute min and max smoothing factors
    smin = N - np.sqrt(2*N)
    smax = N + np.sqrt(2*N)
    smax_limit = 50 * smax

    if plot:
        fig = plt.figure(figsize=(12,9))
        plt.axvline(x[0], color="grey", alpha=0.5)
        plt.axvline(x[-1], color="grey", alpha=0.5)
        dhdx = np.diff(he) / dx
        xc = 0.5 * (xe[1:] + xe[:-1])
        hc = 0.5 * (he[1:] + he[:-1])
        plt.plot(xe, he, "k-", label="raw (%i)" % np.sum(dhdx < seps))
        plt.plot(xc[dhdx < seps], hc[dhdx < seps], "r+")
        plt.legend()
        plt.show()
    
    if plot:
        fig = plt.figure(figsize=(12,9))
        plt.axvline(x[0], color="grey", alpha=0.5)
        plt.axvline(x[-1], color="grey", alpha=0.5)
        dhdx = np.diff(h) / dx
        plt.plot(xe, he, "k-", label="raw (%i)" % np.sum(dhdx < seps))
        for s in [0.5*smin, smin, 0.5*(smin+smax), smax, 2.0*smax]:
            spl = interpolate.splrep(xe, he, k=3, s=s)
            hfilt = interpolate.splev(xe, spl)
            dhfiltdx = np.diff(hfilt) / dx
            plt.plot(xe, hfilt, "--", label="s=%f (%i)" % (s, np.sum(dhfiltdx < seps)))
        plt.legend()
        plt.show()

    # Search optimal smoothing factor
    Ntest = 100
    smin = 0.5 * smin
    smax = 3.0 * smax
    copt = 9999
    while copt > 0:
    
        ci = np.ones(Ntest, dtype=int) * 9999
        si = np.ones(Ntest) * np.nan

        for i in range(0, Ntest):

            s = smin + float(i) / float(Ntest-1) * (smax - smin)
            try:
                spl = interpolate.splrep(xe, he, k=3, s=s)
            except:
                plt.plot(xe, he)
                plt.plot(xe, hfilt)
                plt.show()
            hfilt = interpolate.splev(xe, spl)
            dhfiltdx = np.diff(hfilt) / dx
            si[i] = s
            ci[i] = np.sum(dhfiltdx < seps)
            # print(si[i], ci[i])
            if ci[i] == 0:
                break
        sopt = si[np.isfinite(si)][-1]
        copt = ci[np.isfinite(si)][-1]
        if copt > 0:
            smin = smax
            smax = 2.0 * smax
            if smax > smax_limit:
                print("[WARNING] spline interpolation failed")
                return None

        # else:
        #     raise RuntimeError("Unable to find optimal smoothing parameter")

    # Compute filtered profile
    spl = interpolate.splrep(xe, he, k=3, s=sopt)
    hfilt = interpolate.splev(xe, spl)

    return hfilt[Ne:Ne+N]