import argparse
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import os
import tqdm
import scipy.stats as sps
import sys

from HiVDI.core.pepsi.PepsiNetCDF import PepsiNetCDF
from HiVDI.core.swot.SwotObsReachScale import SwotObsReachScale
from rt_river_segmentation.denoising import hydraulic_filtering
from rt_river_segmentation.segmentation_methods import *

import scipy.stats as sps


def compute_new_case(initial_case, segmentation_bounds):
    
    new_case = initial_case.copy(copy_reaches=False)
    nodes = new_case.nodes
    
    # Special treatment for last node (may be outside segmentation zone)
    if segmentation_bounds[-1] < initial_case.reaches.xbounds[1][-1]:
        print("- Update last reach bounds: %f -> %f" % (segmentation_bounds[-1], initial_case.reaches.xbounds[1][-1]))
        segmentation_bounds[-1] = initial_case.reaches.xbounds[1][-1]
    
    
    # Compute reaches variables
    t = new_case._t
    xbounds = (segmentation_bounds[:-1], segmentation_bounds[1:])
    Hr = np.zeros((new_case._t.size, len(segmentation_bounds)-1))
    Wr = np.zeros((new_case._t.size, len(segmentation_bounds)-1))
    Sr = np.zeros((new_case._t.size, len(segmentation_bounds)-1))
    Qr = np.ones((new_case._t.size, len(segmentation_bounds)-1)) * np.nan
    Ar = np.ones((new_case._t.size, len(segmentation_bounds)-1)) * np.nan
    for ir in range(0, len(segmentation_bounds)-1):
        
        print("- Compute variables for reach %03i/%03i" % (ir+1, len(segmentation_bounds)-1))
        
        if ir == 0:
        
            reach_nodes = np.ravel(np.argwhere(np.logical_and(nodes.x >= segmentation_bounds[ir], 
                                                              nodes.x <= segmentation_bounds[ir+1])))
            
        else:
            
            reach_nodes = np.ravel(np.argwhere(np.logical_and(nodes.x > segmentation_bounds[ir], 
                                                              nodes.x <= segmentation_bounds[ir+1])))
            
        # Update reach index for selected nodes
        nodes._reach[reach_nodes] = ir
            
        # Compute mean values and slope
        x = nodes.x[reach_nodes]
        for it in range(0, new_case._t.size):
            Ht = nodes.H[it, reach_nodes]
            Wt = nodes.W[it, reach_nodes]
            slope, intercept, _, _, _ = sps.linregress(x, Ht)
            Hr[it, ir] = np.mean(Ht)
            Wr[it, ir] = np.mean(Wt)
            Sr[it, ir] = -slope
            if nodes._Q is not None:
                Qt = nodes.Q[it, reach_nodes]
                Qr[it, ir] = np.mean(Qt)
            if nodes._A is not None:
                At = nodes.A[it, reach_nodes]
                Ar[it, ir] = np.mean(At)
                
            
        #plt.plot(nodes.x, np.mean(nodes.H, axis=0))
        #plt.axvline(segmentation_bounds[ir], c="k", ls="--")
        #plt.axvline(segmentation_bounds[ir+1], c="k", ls="--")
        #plt.plot(nodes.x[reach_nodes], np.mean(nodes.H[:, reach_nodes], axis=0), 'r.')
        #plt.plot(0.5 * (xbounds[0][ir] + xbounds[1][ir]), np.mean(Hr[:, ir]), 'b+')
        #plt.show()
        
    
        
    # Create SwotObsReachScale object
    #print(nodes.reach, case + "-Segmented")
    new_case._reaches = SwotObsReachScale(t, xbounds, Hr, Wr, Sr, new_case._time_format, nodes.reach)
    new_case._reaches._Q = Qr
    new_case._reaches._A = Ar
    #print("Ar=", Ar)
    #choice = input()
    
    return new_case
            
                            


def compute_segmentation_and_discharge(case, output, segmentation_method="baseline", lc=5.0, min_length=None):
    
    if os.path.isfile(os.path.expandvars("$PEPSI1_DIR/%s.nc" % case)):
        fname = "$PEPSI1_DIR/%s.nc" % case
    elif os.path.isfile(os.path.expandvars("$PEPSI2_DIR/%s.nc" % case)):
        fname = "$PEPSI2_DIR/%s.nc" % case
    else:
        raise RuntimeError("Case %s not found" % case)
    
    if output is None:
        output = fname.replace("%s.nc" % case, "%s_segmented.nc" % case)
    
    print("*" * 80)
    print(" GENERATE SEGMENTED CASE")
    print("*" * 80)
    print("- case: %s" % case)
    print("- output: %s" % output)
    print("- method: %s" % segmentation_method)
    print("- lambda: %.2f km" % lc)
    print("- min_length: %.2f km" % lc)
    print("*" * 80)
    
    # Load PEPSI case
    print("=" * 80)
    print(" LOAD INITIAL PEPSI CASE SEGMENTED CASE")
    print("=" * 80)
    print("- Load PEPSI data")
    initial_case = PepsiNetCDF(os.path.expandvars(fname), obs_only=False)
    initial_case.compute_true_A0()
    initial_case.nodes.compute_S()
    print("=" * 80)
    
    # Perform segmentation
    print("=" * 80)
    print(" COMPUTE SEGMENTATION")
    print("=" * 80)
    x = initial_case.nodes.x
    #x0 = x[0] - x
    # => x = x[0] - x0
    Z = np.mean(initial_case.nodes.H, axis=0)
    #plt.plot(x, Z)
    #plt.show()
    W = np.mean(initial_case.nodes.W, axis=0)
    
    #plt.plot(x, Z)
    #for rch_bnd in initial_case.reaches._xbounds[0]:
        #plt.axvline(rch_bnd, c="k", ls="--")
    #plt.show()

    
    pas = 10
    N = int(np.floor((x[-1] - x[0]) / pas))
    xi = np.linspace(x[0], x[0] + N * pas, N+1, endpoint=True)
    Zi = np.interp(xi, x, Z)
    Wi = np.interp(xi, x, W)
    #print(x)
    #print(xi)
    #plt.plot(x * 0.001, Z)
    #plt.plot(xi * 0.001, Zi, "--")
    #plt.show()
    
    print("-" * 80)
    print("Hydraulic filtering")
    print("-" * 80)
    Zfiltered = hydraulic_filtering(xi, Zi, x_direction="downstream", plot_steps=False)
    Zi = Zfiltered
    print("-" * 80)

    #plt.plot(x * 0.001, Z)
    #plt.plot(xi * 0.001, Zi, "--")
    #plt.show()
    
    # Convert variables in km to m
    lc = lc * 1000.0
    if min_length is not None:
        min_length = min_length * 1000.0
    
    if segmentation_method == "baseline":
        d2xZ_pos, d2xZ_neg, segmentation_bounds, wavelet = segmentation_baseline(xi, Zi, None, lc)
    else:
        _, _, _, _, segmentation_bounds, wavelet_Z, wavelet_W = segmentation_advanced(xi, Zi, None, Wi, None, lc, min_length=1000.0)

    ## Retransform segmentation bounds to outlet distances
    #segmentation_bounds = x[0] - segmentation_bounds
    
    plt.plot(xi * 0.001, Zi)
    plt.axvline(segmentation_bounds[0] * 0.001, c="r", ls="-", label="segmentation")
    for rch_bnd in segmentation_bounds[1:]:
        plt.axvline(rch_bnd * 0.001, c="r", ls="-")
    for rch_bnd in initial_case.reaches._xbounds[0]:
        plt.axvline(rch_bnd * 0.001, c="k", ls="--")
    plt.axvline(initial_case.reaches._xbounds[1][-1] * 0.001, c="k", ls="--", label="PEPSI reaches")
    plt.xlabel("x (km)")
    plt.ylabel("Z (m)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
    # Recompute data (widths, heights and slope) on new reaches
    new_case = compute_new_case(initial_case, segmentation_bounds)
    new_case.reaches.compute_dA()
    new_case.compute_true_A0()
    
    # Search typical flow lines
    qin = initial_case.reaches.Q[:, 0]
    indices_sorted = np.argsort(qin)
    
    centiles = [10, 20, 50, 80, 90]
    indices_centiles = [indices_sorted[indices_sorted.size * centile // 100]  for centile in centiles]
    
    
    K = 20.0
    for i, index in enumerate(indices_centiles):
        
        # Compute K at nodes scale
        A0n = initial_case.nodes.A0[:]
        dAn = initial_case.nodes.dA[index, :]
        Wn = initial_case.nodes.W[index, :]
        Sn = initial_case.nodes.S[index, :]
        Qn = initial_case.nodes.Q[index, :]
        Kn = Qn / ((A0n + dAn)**(5./3.) * Wn**(-2./3.) * Sn**(0.5))
        Kn[~np.isfinite(Kn)] = np.nan
        Qn2 = Kn * (A0n + dAn)**(5./3.) * Wn**(-2./3.) * Sn**(0.5)
        
        # Computes means of K at reach scale (initial)
        Kr = np.zeros(initial_case.reaches.A0.size)
        A0r2 = np.zeros(initial_case.reaches.A0.size)
        dAr2 = np.zeros(initial_case.reaches.A0.size)
        Wr2 = np.zeros(initial_case.reaches.A0.size)
        Sr2 = np.zeros(initial_case.reaches.A0.size)
        reach_nodes = initial_case.reaches.reach_nodes
        for ir in range(0, initial_case.reaches.A0.size):
            #print(ir, reach_nodes[ir])
            #print("K ", Kn[reach_nodes[ir]])
            Kr[ir] = np.nanmean(Kn[reach_nodes[ir]])
            A0r2[ir] = np.nanmean(A0n[reach_nodes[ir]])
            dAr2[ir] = np.nanmean(dAn[reach_nodes[ir]])
            Wr2[ir] = np.nanmean(Wn[reach_nodes[ir]])
            Sr2[ir] = np.nanmean(Sn[reach_nodes[ir]])
            #print("=> Kr[ir]=", Kr[ir])
            #choice = input()
        
        # Compute Q at reach scale (initial)
        A0r = initial_case.reaches.A0[:]
        dAr = initial_case.reaches.dA[index, :]
        Wr = initial_case.reaches.W[index, :]
        Sr = initial_case.reaches.S[index, :]
        Qr = Kr * (A0r + dAr)**(5./3.) * Wr**(-2./3.) * Sr**(0.5)
        Qr2 = Kr * (A0r2 + dAr2)**(5./3.) * Wr2**(-2./3.) * Sr2**(0.5)
        
        bias = np.mean(Qr) - np.mean(Qn)
        Qr -= bias
        
        #xr = np.concatenate((initial_case.reaches.xbounds[0], [initial_case.reaches.xbounds[1][-1]]))
        #fig, axes = plt.subplots(6, 1, sharex=True)
        
        #axes[0].plot(initial_case.nodes.x, Qn, 'r.')
        #axes[0].plot(initial_case.nodes.x, Qn2, 'g--')
        #axes[0].step(xr, np.concatenate((Qr, [Qr[-1]])), 'b-', where='post')
        #axes[0].plot(xr, np.concatenate((Qr2, [Qr2[-1]])), 'g-')

        #axes[1].plot(initial_case.nodes.x, Sn, 'r.')
        #axes[1].step(xr, np.concatenate((Sr, [Sr[-1]])), 'b-', where='post')
        #axes[1].step(xr, np.concatenate((Sr2, [Sr2[-1]])), 'g-', where='post')

        #axes[2].plot(initial_case.nodes.x, A0n, 'r.')
        #axes[2].step(xr, np.concatenate((A0r, [A0r[-1]])), 'b-', where='post')
        #axes[2].step(xr, np.concatenate((A0r2, [A0r2[-1]])), 'g-', where='post')

        #axes[3].plot(initial_case.nodes.x, dAn, 'r.')
        #axes[3].step(xr, np.concatenate((dAr, [dAr[-1]])), 'b-', where='post')
        #axes[3].step(xr, np.concatenate((dAr2, [dAr2[-1]])), 'g-', where='post')

        #axes[4].plot(initial_case.nodes.x, Wn, 'r.')
        #axes[4].step(xr, np.concatenate((Wr, [Wr[-1]])), 'b-', where='post')
        #axes[4].step(xr, np.concatenate((Wr2, [Wr2[-1]])), 'g-', where='post')

        #axes[5].plot(initial_case.nodes.x, Kn, 'r.')
        #axes[5].step(xr, np.concatenate((Kr, [Kr[-1]])), 'b-', where='post')

        #plt.show()
        full_figure = True
        if full_figure:
            fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8,12))
            axes[0].plot(initial_case.nodes.x * 0.001, Qn, 'r.', label=r"$Q_{true}$ (nodes)")
            axes[1].plot(initial_case.nodes.x * 0.001, A0n, 'r.')
            axes[2].plot(initial_case.nodes.x * 0.001, Wn, 'r.')
            axes[3].plot(initial_case.nodes.x * 0.001, Sn, 'r.')
        
        xr = np.concatenate((initial_case.reaches.xbounds[0], [initial_case.reaches.xbounds[1][-1]]))
        if full_figure:
            axes[0].step(xr * 0.001, np.concatenate((Qr, [Qr[-1]])), 'b-', where='post', label=r"Reaches PEPSI")
            axes[1].step(xr * 0.001, np.concatenate((A0r, [A0r[-1]])), 'b-', where='post')
            axes[2].step(xr * 0.001, np.concatenate((Wr, [Wr[-1]])), 'b-', where='post')
            axes[3].step(xr * 0.001, np.concatenate((Sr, [Sr[-1]])), 'b-', where='post')
        else:
            plt.fill_between(initial_case.nodes.x * 0.001, 0.7 * Qn, 1.3 * Qn, color='r', alpha=0.5)
            plt.plot(initial_case.nodes.x * 0.001, Qn, 'r.', label=r"$Q_{true}$ (nodes)")
            plt.step(xr * 0.001, np.concatenate((Qr, [Qr[-1]])), 'b-', where='post', label=r"$Q$ segmentation initiale")
        
        nr = len(initial_case.reaches.xbounds[0])
        lr = initial_case.reaches.xbounds[1] - initial_case.reaches.xbounds[0]
        print("- Initial segmentation : %03i reaches, [%.2f - %.2f]" % (nr, np.min(lr), np.max(lr)))
        
        # Computes means of K at reach scale (segmentation)
        Kr = np.zeros(new_case.reaches.A0.size)
        reach_nodes = new_case.reaches.reach_nodes
        for ir in range(0, new_case.reaches.A0.size):
            Kr[ir] = np.nanmean(Kn[reach_nodes[ir]])
        
        # Compute Q at reach scale
        A0r = new_case.reaches.A0[:]
        dAr = new_case.reaches.dA[index, :]
        Wr = new_case.reaches.W[index, :]
        Sr = new_case.reaches.S[index, :]
        Qr = Kr * (A0r + dAr)**(5./3.) * Wr**(-2./3.) * Sr**(0.5)
        bias = np.nanmean(Qr) - np.mean(Qn)
        Qr -= bias
        print("A0=", A0r)
        print("dA=", dAr)
        print("W=", Wr)
        print("S=", Sr)
        print("K=", Kr)
        print("bias=", bias)
        print(Qr)
        xr = np.concatenate((new_case.reaches.xbounds[0], [new_case.reaches.xbounds[1][-1]]))
        nr = len(new_case.reaches.xbounds[0])
        lr = new_case.reaches.xbounds[1] - new_case.reaches.xbounds[0]
        print("- New segmentation : %03i reaches, [%.2f - %.2f]" % (nr, np.min(lr), np.max(lr)))
        if full_figure:
            axes[0].step(xr * 0.001, np.concatenate((Qr, [Qr[-1]])), 'g--', where='post', label=r"Reaches segmentation")
            axes[1].step(xr * 0.001, np.concatenate((A0r, [A0r[-1]])), 'g--', where='post')
            axes[2].step(xr * 0.001, np.concatenate((Wr, [Wr[-1]])), 'g--', where='post')
            axes[3].step(xr * 0.001, np.concatenate((Sr, [Sr[-1]])), 'g--', where='post')
        else:
            plt.step(xr * 0.001, np.concatenate((Qr, [Qr[-1]])), 'g--', where='post', label=r"$Q$ nouvelle segmentation")
            plt.legend()
            plt.xlabel("x (km)")
            plt.ylabel("Q (m3/s)")
            plt.title(r"$Q_{%i}$" % centiles[i])
            plt.tight_layout()
            plt.show()

        if full_figure:
            axes[0].set_ylim(0.0, axes[0].get_ylim()[1])
            axes[0].legend(loc="lower right")
            axes[0].set_ylabel("Q (m3/s)")
            axes[1].set_ylabel("A0 (m2)")
            axes[2].set_ylabel("W (m)")
            axes[3].set_ylabel("S (m/m)")
            axes[3].set_xlabel("x (km)")
            #fig.suptitle(r"$Q_{%i}$" % centiles[i])
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Test the estimation of discharge using the Low-Froude model on a segmentation of a PEPSI case")
    parser.add_argument("case", type=str, help="PEPSI case")
    parser.add_argument("-segmentation", dest="segmentation_method", type=str, choices=["baseline", "advanced"], 
                        default="baseline", help="Segmentation method")
    parser.add_argument("-lambda", dest="lambdac", type=float, default=5.0,
                        help="Segmentation characteristic length in km")
    parser.add_argument("-min-length", dest="min_length", type=float, default=None,
                        help="Minimal reach length in km")
    args = parser.parse_args()
    
    compute_segmentation_and_discharge(args.case, args.output, args.segmentation_method, args.lambdac, args.min_length)
