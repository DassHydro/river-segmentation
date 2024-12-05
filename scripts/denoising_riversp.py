#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kevin Larnier (kevin.larnier@hydro-matters.fr)
"""

import argparse
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as sps

from rt_river_segmentation.denoising import hydraulic_filtering
from rt_river_segmentation.filtering_splines import hydraulic_preserving_spline
# from rt_river_segmentation.load_and_process_data import *
# from rt_river_segmentation.segmentation_methods import segmentation_baseline, segmentation_advanced


def remove_outliers_and_denoising(xn, Hn, Wn, diff_crit1=20.0, diff_crit2=10.0, dx=10.0, plot=True):

    xn = xn[::-1]
    Hn = Hn[::-1]
    Wn = Wn[::-1]
    xn0 = xn.copy()
    Hn0 = Hn.copy()
    Wn0 = Wn.copy()

    indices = np.arange(0, xn.size)
    valid_indices = np.arange(0, xn.size)
    removed_count = 0

    # Remove outliers using difference from mean slope line
    diffmax = 9999.0
    #removed_count = 0
    while diffmax > diff_crit1:
        slp, H0, _, _, _ = sps.linregress(xn, Hn)
        #print(slp)
        if slp < 1e-12:
            H0 = np.mean(Hn)
            slp = 0.0
        #if slp
        Hns = H0 + slp * xn
        diff = np.abs(Hn - Hns)
        diffmax = np.max(diff)
        diffmax_index = np.argmax(diff)
        xn = np.delete(xn, diffmax_index)
        Hn = np.delete(Hn, diffmax_index)
        Wn = np.delete(Hn, diffmax_index)
        Hns = H0 + slp * xn
        valid_indices = np.delete(valid_indices, diffmax_index)
        removed_count += 1
    removed_count_pass1 = removed_count
    outliers1_indices = indices[~np.isin(indices, valid_indices)]
    print("     PASS1: %i removed" % removed_count_pass1)

    # Remove outliers using difference from cubic spline
    diffmax = 9999.0
    Hf = None
    while diffmax > diff_crit2:

        rxn = (xn - xn[-1])[::-1]
        rHn = Hn[::-1]
        N = int(np.floor(rxn[-1]/50.0))
        rxi = np.linspace(0, N * 50.0, N+1, endpoint=True)
        rHi = np.interp(rxi, rxn, rHn)
        rHf = hydraulic_preserving_spline(rxi, rHi, dx=50.0, seps=1e-9, le_min=20000.0)
        if rHf is None:
            break
        Hf = np.interp(rxn, rxi, rHf)[::-1]
        diff = np.abs(Hn - Hf)
        diffmax = np.max(diff)
        diffmax_index = np.argmax(diff)
        xn = np.delete(xn, diffmax_index)
        Hn = np.delete(Hn, diffmax_index)
        Hf = np.delete(Hf, diffmax_index)
        valid_indices = np.delete(valid_indices, diffmax_index)
        removed_count += 1
    removed_count_pass2 = removed_count - removed_count_pass1
    outliers2_indices = indices[np.logical_and(~np.isin(indices, valid_indices), ~np.isin(indices, outliers1_indices))]
    print("     PASS2: %i removed" % removed_count_pass2)
    print("     FINAL: %i removed" % removed_count)

    if valid_indices.size > 0:
        xn = xn[::-1]
        Hn = Hn[::-1]
        Wn = Wn[::-1]
        N = (xn[-1] - xn[0]) / dx
        xi = xn[0] + np.arange(0, N) * dx
        Hi = np.interp(xi, xn, Hn)
        Hfilt = hydraulic_filtering(xi, Hi, x_direction="downstream", plot_steps=False)
        Hfilt = np.interp(xn0, xi, Hfilt)
    else:
        Hfilt = None


    return xn0[::-1], Hfilt[::-1], Wn0[::-1], {"valid":xn0.size-1-valid_indices, "outliers_pass1": xn0.size-1-outliers1_indices, "outliers_pass2": xn0.size-1-outliers2_indices}


def load_riversp(riversp_file, upstream_reachid, downstream_reach_id, dx=50):

    if "Reach" in os.path.basename(riversp_file):
        reach_file = riversp_file
        node_file = os.path.join(os.path.dirname(riversp_file), os.path.basename(riversp_file).replace("Reach", "Node"))
    else:
        node_file = riversp_file
        reach_file = os.path.join(os.path.dirname(riversp_file), os.path.basename(riversp_file).replace("Node", "Reach"))


    if not os.path.isfile(reach_file):
        raise IOError("RiverSP Reach file not found: %s" % reach_file)
    if not os.path.isfile(node_file):
        raise IOError("RiverSP Node file not found: %s" % node_file)

    # Read reach_file and retrieve list of reaches
    print("Load RiverSP Reach file...")
    reaches_list = []
    riversp_reach = gpd.read_file(reach_file)
    upstream_reach_row = riversp_reach[riversp_reach["reach_id"].astype(int) == upstream_reachid]
    if upstream_reach_row.index.size == 0:
        raise RuntimeError("Upstream reach with reachid=%i not found" % upstream_reachid)
    current_reach_row = upstream_reach_row
    while int(current_reach_row.loc[current_reach_row.index[0], "reach_id"]) != downstream_reach_id:
        reaches_list.append(int(current_reach_row.loc[current_reach_row.index[0], "reach_id"]))
        reach_id = int(current_reach_row.loc[current_reach_row.index[0], "reach_id"])
        n_rch_dn = current_reach_row.loc[current_reach_row.index[0], "n_reach_dn"]
        if n_rch_dn < 1:
            raise RuntimeError("Reach with reach_id=%i has not downstream reach" % reach_id)
        if n_rch_dn > 1:
            raise RuntimeError("Reach with reach_id=%i has multiple downstream reaches" % reach_id)
        reach_id_dn = int(current_reach_row.loc[current_reach_row.index[0], "rch_id_dn"].split(",")[0])
        current_reach_row = riversp_reach[riversp_reach["reach_id"].astype(int) == reach_id_dn]
    reaches_list.append(downstream_reach_id)
    print("- Reaches list: [%s]" % ",".join(map(str, reaches_list)))
    print("")

    # Load node file
    print("Load RiverSP Node file...")
    riversp_nodes = gpd.read_file(node_file)
    riversp_nodes = riversp_nodes[riversp_nodes["reach_id"].astype(int).isin(reaches_list)]
    riversp_nodes = riversp_nodes.sort_values(by=["p_dist_out"], ascending=True)
    print("- Number of nodes: %i" % riversp_nodes.index.size)

    # Resample data
    x = riversp_nodes["p_dist_out"].values
    wse = riversp_nodes["wse"].values
    width = riversp_nodes["width"].values

    valid = np.logical_and(wse > -999999.0, riversp_nodes["dark_frac"].values < 0.5)    
    x = x[valid]
    width = width[valid]
    wse = wse[valid]
    print("- Number of valid nodes: %i" % wse.size)
    print("")

    if dx is not None:
        N = int(np.floor(x[-1] - x[0]) / dx)
        xi = np.linspace(x[0], x[0]+N*dx, N+1, endpoint=True)
        wsei = np.interp(xi, x, wse)
        widthi = np.interp(xi, x, width)
    else:
        xi, wsei, widthi = x, wse, width

    return xi, wsei, widthi


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Procedure to realize a segmentation of a signal of wse from a riversp file")
    parser.add_argument("riversp_file", type=str, help="path to riversp file")
    parser.add_argument("upstream_reachid", type=int, help="ID of the upstream reach")
    parser.add_argument("downstream_reachid", type=int, help="ID of the downstream reach")
    parser.add_argument("-dx", type=int, default=10, help="Spacing to resample data (m)")
    parser.add_argument("-csv-file", dest="csv_file", type=str, default=None, help="Path to the csv output file")
    parser.add_argument("-plot-file", dest="plot_file", type=str, default=None, help="Path to the plot file")
                        
    args = parser.parse_args()

    # Load RiverSP files
    x, H, W = load_riversp(args.riversp_file, args.upstream_reachid, args.downstream_reachid, dx=None)

    # PROCESSING
    xf, Hf, Wf, indices = remove_outliers_and_denoising(x, H, W)

    # Create CSV
    if args.csv_file is not None:
        outlier_flag = np.zeros(x.size, dtype=int)
        outlier_flag[indices["outliers_pass1"]] = 1
        outlier_flag[indices["outliers_pass2"]] = 2
        dataframe = pd.DataFrame(data={"x": x, "H": H, "Hfilt": Hf, "outlier_flag": outlier_flag})
        dataframe.to_csv(args.csv_file, sep=";", index=False)

    # Make plot
    plt.plot(x[indices["valid"]] * 0.001, H[indices["valid"]], "g.")
    plt.plot(x[indices["outliers_pass1"]] * 0.001, H[indices["outliers_pass1"]], "ro", label="outliers (pass1)")
    plt.plot(x[indices["outliers_pass2"]] * 0.001, H[indices["outliers_pass2"]], c="orange", marker="o", label="outliers (pass2)")
    plt.plot(xf * 0.001, Hf, c="blue", ls="--", label="filtered")
    plt.legend()
    plt.ylabel("wse (m)")
    plt.xlabel("outlet distance (km)")
    if args.plot_file is not None:
        plt.savefig(args.plot_file)
    else:
        plt.show()
    
    # if args.denoising:
        
    #     #--------------------------------------------------------------------------------------------------------------
    #     # Use hydraulic based filtering method
    #     #--------------------------------------------------------------------------------------------------------------

    #     print("\n" + "-" * 80)
    #     print("Hydraulic based filtering")
    #     print("-" * 80)
    #     Zfiltered = hydraulic_filtering(X, Z[::-1], x_direction="downstream") 
    #     Z = Zfiltered[::-1]
    #     print("-" * 80 + "\n")
    
    # if not args.segment_widths:
        
    #     #--------------------------------------------------------------------------------------------------------------
    #     # Segmentation "baseline" (using heights only)
    #     #--------------------------------------------------------------------------------------------------------------
        
    #     # Compute segmentations for every lambda_c
    #     wavelet = None
    #     d2xZ_pos_list = []
    #     d2xZ_neg_list = []
    #     for lambda_c in lambda_c_list:
    #         if wavelet is None:
    #             d2xZ_pos, d2xZ_neg, reaches_bounds, wavelet = segmentation_baseline(X, Z, None, lambda_c)
    #         else:
    #             d2xZ_pos, d2xZ_neg, reaches_bounds, _ = segmentation_baseline(X, Z, None, lambda_c, wavelet)
    #         d2xZ_pos_list.append(d2xZ_pos)
    #         d2xZ_neg_list.append(d2xZ_neg)

    #     # Create plot
    #     figure = plt.figure(figsize = (12, 8))
    #     offset = 0.0
    #     for index, lambda_c in enumerate(lambda_c_list):
            
    #         if lambda_c < 1000:
    #             lambda_with_unit = "%.1f m" % lambda_c
    #         else:
    #             lambda_with_unit = "%.1f km" % (lambda_c/1000.0)
            
    #         d2xZ_pos = d2xZ_pos_list[index]
    #         d2xZ_neg = d2xZ_neg_list[index]
    #         if index == 0:
    #             plt.plot(d2xZ_pos[0]/1000, d2xZ_pos[1], linewidth = 4, label = r'$Z(\partial_x^2 > 0)$', color = 'g', linestyle = '-')
    #             plt.plot(d2xZ_neg[0]/1000, d2xZ_neg[1], linewidth = 4, label = r'$Z(\partial_x^2 < 0)$', color = 'r', linestyle = '-')
    #         else:
    #             plt.plot(d2xZ_pos[0]/1000, d2xZ_pos[1] + offset, linewidth = 4, color = 'g', linestyle = '-')
    #             plt.plot(d2xZ_neg[0]/1000, d2xZ_neg[1] + offset, linewidth = 4, color = 'r', linestyle = '-')
    #         if len(lambda_c_list) > 1:
    #             plt.text(X[-1]/1000, Z[-1] + offset, r"$\lambda_c=%s$" % lambda_with_unit)
    #         offset += 3
    #     plt.xlabel("xs (km)")
    #     plt.ylabel("Z (m)")
        
    #     if len(lambda_c_list) > 1:
    #         plt.title(r"Segmentation with various $\lambda_c$ values")
    #     else:
    #         lambda_c = lambda_c_list[0]
    #         if lambda_c < 1000:
    #             lambda_with_unit = "%.1f m" % lambda_c
    #         else:
    #             lambda_with_unit = "%.1f km" % (lambda_c/1000.0)
    #         plt.title(r"Segmentation with $\lambda_c$=%s" % lambda_with_unit)
    #     plt.legend()
    #     if args.plot_file is not None:
    #         plt.savefig(args.plot_file)
    #     else:
    #         plt.show()
        
    # else:
    
    #     #--------------------------------------------------------------------------------------------------------------
    #     # Segmentation "advanced" (using heights and widths)
    #     #--------------------------------------------------------------------------------------------------------------

    #     # Compute segmentations for every lambda_c
    #     wavelet_Z = None
    #     wavelet_W = None
    #     d2xZ_ZpWp_list = []
    #     d2xZ_ZpWn_list = []
    #     d2xZ_ZnWp_list = []
    #     d2xZ_ZnWn_list = []
    #     for lambda_c in lambda_c_list:
    #         if wavelet_Z is None:
    #             results = segmentation_advanced(X, Z, Zb, W, Ah, lambda_c, 
    #                                             add_max_curvature_W=args.enable_width_max_curvature_points)
    #             d2xZ_ZpWp, d2xZ_ZpWn, d2xZ_ZnWp, d2xZ_ZnWn, reaches_bounds, wavelet_Z, wavelet_W = results
    #         else:
    #             results = segmentation_advanced(X, Z, Zb, W, Ah, lambda_c, wavelet_Z, wavelet_W,
    #                                             add_max_curvature_W=args.enable_width_max_curvature_points)
    #             d2xZ_ZpWp, d2xZ_ZpWn, d2xZ_ZnWp, d2xZ_ZnWn, reaches_bounds, _, _ = results
    #         d2xZ_ZpWp_list.append(d2xZ_ZpWp)
    #         d2xZ_ZpWn_list.append(d2xZ_ZpWn)
    #         d2xZ_ZnWp_list.append(d2xZ_ZnWp)
    #         d2xZ_ZnWn_list.append(d2xZ_ZnWn)

    #     # Create plot
    #     figure, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 8), gridspec_kw={'height_ratios': [3, 1]})
    #     offset = 0.0
    #     for index, lambda_c in enumerate(lambda_c_list):
            
    #         if lambda_c < 1000:
    #             lambda_with_unit = "%.1f m" % lambda_c
    #         else:
    #             lambda_with_unit = "%.1f km" % (lambda_c/1000.0)
            
    #         d2xZ_ZpWp = d2xZ_ZpWp_list[index]
    #         d2xZ_ZpWn = d2xZ_ZpWn_list[index]
    #         d2xZ_ZnWp = d2xZ_ZnWp_list[index]
    #         d2xZ_ZnWn = d2xZ_ZnWn_list[index]
            
    #         if index == 0:

    #             ax1.plot(d2xZ_ZpWp[0]/1000, d2xZ_ZpWp[1], linewidth = 4, label = r'$Z(\partial_x^2 > 0), W(\partial_x^2 > 0)$', color = 'g', linestyle = '-')
    #             ax1.plot(d2xZ_ZpWn[0]/1000, d2xZ_ZpWn[1], linewidth = 4, label = r'$Z(\partial_x^2 > 0), W(\partial_x^2 < 0)$', color = 'r', linestyle = '-')
    #             ax1.plot(d2xZ_ZnWp[0]/1000, d2xZ_ZnWp[1], linewidth = 4, label = r'$Z(\partial_x^2 < 0), W(\partial_x^2 > 0)$', color = 'b', linestyle = '-')
    #             ax1.plot(d2xZ_ZnWn[0]/1000, d2xZ_ZnWn[1], linewidth = 4, label = r'$Z(\partial_x^2 < 0), W(\partial_x^2 < 0)$', color = 'y', linestyle = '-')
                
    #         else:
    #             ax1.plot(d2xZ_ZpWp[0]/1000, d2xZ_ZpWp[1]+offset, linewidth = 4, color = 'g', linestyle = '-')
    #             ax1.plot(d2xZ_ZpWn[0]/1000, d2xZ_ZpWn[1]+offset, linewidth = 4, color = 'r', linestyle = '-')
    #             ax1.plot(d2xZ_ZnWp[0]/1000, d2xZ_ZnWp[1]+offset, linewidth = 4, color = 'b', linestyle = '-')
    #             ax1.plot(d2xZ_ZnWn[0]/1000, d2xZ_ZnWn[1]+offset, linewidth = 4, color = 'y', linestyle = '-')
    #         if len(lambda_c_list) > 1:
    #             ax1.text(X[-1]/1000, Z[-1] + offset, r"$\lambda_c=%s$" % lambda_with_unit)
    #         offset += 3
    #     ax1.set_xlabel("xs (km)")
    #     ax1.set_ylabel("Z(m)")
    #     ax1.legend()
    #     ax2.plot(X/1000, W, "k-", label = "W")
    #     ax2.set_xlabel("xs (km)")
    #     ax2.set_ylabel("W (m)")
    #     figure.suptitle(r"Segmentation with various $\lambda_c$ values")
    #     figure.tight_layout(rect=[0, 0, 1, 0.95])
    #     if args.plot_file is not None:
    #         plt.savefig(args.plot_file)
    #     else:
    #         plt.show()
    
        
    
    
