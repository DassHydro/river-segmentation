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
    print("TEST last reach bounds:", segmentation_bounds[-1], initial_case.reaches.xbounds[1][-1])
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
    for ir in range(0, len(segmentation_bounds)-1):
        
        print("- Compute variables for reach %03i/%03i" % (ir+1, len(segmentation_bounds)-1))
        
        if ir == 0:
        
            reach_nodes = np.ravel(np.argwhere(np.logical_and(nodes.x >= segmentation_bounds[ir], 
                                                              nodes.x <= segmentation_bounds[ir+1])))
            
        else:
            
            reach_nodes = np.ravel(np.argwhere(np.logical_and(nodes.x > segmentation_bounds[ir], 
                                                              nodes.x <= segmentation_bounds[ir+1])))
        print("reach %02i, nodes=" % (ir+1), reach_nodes)
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
                
            
        plt.plot(nodes.x, np.mean(nodes.H, axis=0))
        plt.axvline(segmentation_bounds[ir], c="k", ls="--")
        plt.axvline(segmentation_bounds[ir+1], c="k", ls="--")
        plt.plot(nodes.x[reach_nodes], np.mean(nodes.H[:, reach_nodes], axis=0), 'r.')
        plt.plot(0.5 * (xbounds[0][ir] + xbounds[1][ir]), np.mean(Hr[:, ir]), 'b+')
        plt.show()
        
    
        
    # Create SwotObsReachScale object
    #print(nodes.reach, case + "-Segmented")
    new_case._reaches = SwotObsReachScale(t, xbounds, Hr, Wr, Sr, new_case._time_format, nodes.reach)
    new_case._reaches._Q = Qr
    
    return new_case
            
                            


def generate_case(case, output, segmentation_method="baseline", lc=5.0, min_length=None):
    
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
    print("=" * 80)
    
    # Perform segmentation
    print("=" * 80)
    print(" COMPUTE SEGMENTATION")
    print("=" * 80)

    x = initial_case.nodes.x
    Z = np.mean(initial_case.nodes.H, axis=0)
    W = np.mean(initial_case.nodes.W, axis=0)
    
    pas = 25
    N = int(np.floor((x[-1] - x[0]) / pas))
    xi = np.linspace(x[0], x[0] + N * pas, N+1, endpoint=True)
    Zi = np.interp(xi, x, Z)
    Wi = np.interp(xi, x, W)
    print(x)
    print(xi)
    plt.plot(x * 0.001, Z)
    plt.plot(xi * 0.001, Zi, "--")
    plt.show()
    
    print("-" * 80)
    print("Hydraulic filtering")
    print("-" * 80)
    Zfiltered = hydraulic_filtering(xi, Zi, x_direction="downstream", plot_steps=False)
    Zi = Zfiltered
    print("-" * 80)

    plt.plot(x * 0.001, Z)
    plt.plot(xi * 0.001, Zi, "--")
    plt.show()
    
    # Convert variables in km to m
    lc = lc * 1000.0
    if min_length is not None:
        min_length = min_length * 1000.0
    
    if segmentation_method == "baseline":
        d2xZ_pos, d2xZ_neg, segmentation_bounds, wavelet = segmentation_baseline(xi, Zi, None, lc)
    else:
        _, _, _, _, segmentation_bounds, wavelet_Z, wavelet_W = segmentation_advanced(xi, Zi, None, Wi, None, lc, min_length=1000.0)

    plt.plot(x * 0.001, Z, "b.")
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
    new_case.write(output, case+"-Segmented")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Generate a pseudo PEPSI from a segmented PEPSI original case")
    parser.add_argument("case", type=str, help="PEPSI case")
    parser.add_argument("-o", dest="output", type=str, default=None,
                        help="Path to the output file (default is [CASE_DIR]/[CASE]_segmented.nc)")
    parser.add_argument("-segmentation", dest="segmentation_method", type=str, choices=["baseline", "advanced"], 
                        default="baseline", help="Segmentation method")
    parser.add_argument("-lambda", dest="lambdac", type=float, default=5.0,
                        help="Segmentation characteristic length in km")
    parser.add_argument("-min-length", dest="min_length", type=float, default=None,
                        help="Minimal reach length in km")
    args = parser.parse_args()
    
    generate_case(args.case, args.output, args.segmentation_method, args.lambdac, args.min_length)
