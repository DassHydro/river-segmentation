#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kevin Larnier (kevin.larnier@hydro-matters.fr)
"""

import argparse
import os
import geopandas as gpd

from rt_river_segmentation.denoising import hydraulic_filtering
from rt_river_segmentation.load_and_process_data import *
from rt_river_segmentation.segmentation_methods import *


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
    valid = np.logical_and(valid, riversp_nodes["node_q"].values < 2)    
    # valid = np.logical_and()
    x = x[valid]
    width = width[valid]
    wse = wse[valid]
    print("- Number of valid nodes: %i" % wse.size)
    print("")

    N = int(np.floor(x[-1] - x[0]) / dx)
    xi = np.linspace(x[0], x[0]+N*dx, N+1, endpoint=True)
    wsei = np.interp(xi, x, wse)
    widthi = np.interp(xi, x, width)

    return xi, wsei, widthi


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Procedure to realize a segmentation of a signal of wse from a riversp file")
    parser.add_argument("riversp_file", type=str, help="path to riversp file")
    parser.add_argument("upstream_reachid", type=int, help="ID of the upstream reach")
    parser.add_argument("downstream_reachid", type=int, help="ID of the downstream reach")
    parser.add_argument("-lambda", dest="lambda_c", type=float, nargs="+", default=1.0, help="Caracteristic length(s) for the segmentation (km)")
    parser.add_argument("-dx", type=int, default=10, help="Spacing to resample data (m)")
    parser.add_argument("-plot-file", dest="plot_file", type=str, default=None, help="Path to the plot file")
    parser.add_argument("-segment-widths", dest="segment_widths", action="store_true", help="Use widths for segmentation")
    parser.add_argument("--denoising", action="store_true", 
                        help="Activate hydraulic filtering prior to segmentation")
    parser.add_argument("--enable-width-max-curvature-points", dest="enable_width_max_curvature_points",
                        action="store_true", 
                        help="Enable maximum of curvature of W for segments definitions ('advanced' only)")
                        
    args = parser.parse_args()

    # Load RiverSP files
    X, Z, W = load_riversp(args.riversp_file, args.upstream_reachid, args.downstream_reachid, args.dx)

    plt.plot(X, Z)
    plt.show()

    if isinstance(args.lambda_c, list):
        lambda_c_list = [lambda_c * 1000.0 for lambda_c in args.lambda_c]
    else:
        lambda_c_list = [args.lambda_c * 1000.0]
    
    
    # # PROCESSING
    # X, W, M_Z, Zb, H, dxZ, dx2Z, dxZb, Q, Ah, Ph = load_and_process_data(args.data_files, args.complementary_data, pas)
    # Z = M_Z[time_index, :]
    # W = W[time_index, :]
    # Ah = Ah[time_index, :]
    
    if args.denoising:
        
        #--------------------------------------------------------------------------------------------------------------
        # Use hydraulic based filtering method
        #--------------------------------------------------------------------------------------------------------------

        print("\n" + "-" * 80)
        print("Hydraulic based filtering")
        print("-" * 80)
        Zfiltered = hydraulic_filtering(X, Z[::-1], x_direction="downstream") 
        Z = Zfiltered[::-1]
        print("-" * 80 + "\n")
    
    if not args.segment_widths:
        
        #--------------------------------------------------------------------------------------------------------------
        # Segmentation "baseline" (using heights only)
        #--------------------------------------------------------------------------------------------------------------
        
        # Compute segmentations for every lambda_c
        wavelet = None
        d2xZ_pos_list = []
        d2xZ_neg_list = []
        for lambda_c in lambda_c_list:
            if wavelet is None:
                d2xZ_pos, d2xZ_neg, reaches_bounds, wavelet = segmentation_baseline(X, Z, None, lambda_c)
            else:
                d2xZ_pos, d2xZ_neg, reaches_bounds, _ = segmentation_baseline(X, Z, None, lambda_c, wavelet)
            d2xZ_pos_list.append(d2xZ_pos)
            d2xZ_neg_list.append(d2xZ_neg)

        # Create plot
        figure = plt.figure(figsize = (12, 8))
        offset = 0.0
        for index, lambda_c in enumerate(lambda_c_list):
            
            if lambda_c < 1000:
                lambda_with_unit = "%.1f m" % lambda_c
            else:
                lambda_with_unit = "%.1f km" % (lambda_c/1000.0)
            
            d2xZ_pos = d2xZ_pos_list[index]
            d2xZ_neg = d2xZ_neg_list[index]
            if index == 0:
                plt.plot(d2xZ_pos[0]/1000, d2xZ_pos[1], linewidth = 4, label = r'$Z(\partial_x^2 > 0)$', color = 'g', linestyle = '-')
                plt.plot(d2xZ_neg[0]/1000, d2xZ_neg[1], linewidth = 4, label = r'$Z(\partial_x^2 < 0)$', color = 'r', linestyle = '-')
            else:
                plt.plot(d2xZ_pos[0]/1000, d2xZ_pos[1] + offset, linewidth = 4, color = 'g', linestyle = '-')
                plt.plot(d2xZ_neg[0]/1000, d2xZ_neg[1] + offset, linewidth = 4, color = 'r', linestyle = '-')
            if len(lambda_c_list) > 1:
                plt.text(X[-1]/1000, Z[-1] + offset, r"$\lambda_c=%s$" % lambda_with_unit)
            offset += 3
        plt.xlabel("xs (km)")
        plt.ylabel("Z (m)")
        
        if len(lambda_c_list) > 1:
            plt.title(r"Segmentation with various $\lambda_c$ values")
        else:
            lambda_c = lambda_c_list[0]
            if lambda_c < 1000:
                lambda_with_unit = "%.1f m" % lambda_c
            else:
                lambda_with_unit = "%.1f km" % (lambda_c/1000.0)
            plt.title(r"Segmentation with $\lambda_c$=%s" % lambda_with_unit)
        plt.legend()
        if args.plot_file is not None:
            plt.savefig(args.plot_file)
        else:
            plt.show()
        
    else:
    
        #--------------------------------------------------------------------------------------------------------------
        # Segmentation "advanced" (using heights and widths)
        #--------------------------------------------------------------------------------------------------------------

        # Compute segmentations for every lambda_c
        wavelet_Z = None
        wavelet_W = None
        d2xZ_ZpWp_list = []
        d2xZ_ZpWn_list = []
        d2xZ_ZnWp_list = []
        d2xZ_ZnWn_list = []
        for lambda_c in lambda_c_list:
            if wavelet_Z is None:
                results = segmentation_advanced(X, Z, Zb, W, Ah, lambda_c, 
                                                add_max_curvature_W=args.enable_width_max_curvature_points)
                d2xZ_ZpWp, d2xZ_ZpWn, d2xZ_ZnWp, d2xZ_ZnWn, reaches_bounds, wavelet_Z, wavelet_W = results
            else:
                results = segmentation_advanced(X, Z, Zb, W, Ah, lambda_c, wavelet_Z, wavelet_W,
                                                add_max_curvature_W=args.enable_width_max_curvature_points)
                d2xZ_ZpWp, d2xZ_ZpWn, d2xZ_ZnWp, d2xZ_ZnWn, reaches_bounds, _, _ = results
            d2xZ_ZpWp_list.append(d2xZ_ZpWp)
            d2xZ_ZpWn_list.append(d2xZ_ZpWn)
            d2xZ_ZnWp_list.append(d2xZ_ZnWp)
            d2xZ_ZnWn_list.append(d2xZ_ZnWn)

        # Create plot
        figure, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 8), gridspec_kw={'height_ratios': [3, 1]})
        offset = 0.0
        for index, lambda_c in enumerate(lambda_c_list):
            
            if lambda_c < 1000:
                lambda_with_unit = "%.1f m" % lambda_c
            else:
                lambda_with_unit = "%.1f km" % (lambda_c/1000.0)
            
            d2xZ_ZpWp = d2xZ_ZpWp_list[index]
            d2xZ_ZpWn = d2xZ_ZpWn_list[index]
            d2xZ_ZnWp = d2xZ_ZnWp_list[index]
            d2xZ_ZnWn = d2xZ_ZnWn_list[index]
            
            if index == 0:

                ax1.plot(d2xZ_ZpWp[0]/1000, d2xZ_ZpWp[1], linewidth = 4, label = r'$Z(\partial_x^2 > 0), W(\partial_x^2 > 0)$', color = 'g', linestyle = '-')
                ax1.plot(d2xZ_ZpWn[0]/1000, d2xZ_ZpWn[1], linewidth = 4, label = r'$Z(\partial_x^2 > 0), W(\partial_x^2 < 0)$', color = 'r', linestyle = '-')
                ax1.plot(d2xZ_ZnWp[0]/1000, d2xZ_ZnWp[1], linewidth = 4, label = r'$Z(\partial_x^2 < 0), W(\partial_x^2 > 0)$', color = 'b', linestyle = '-')
                ax1.plot(d2xZ_ZnWn[0]/1000, d2xZ_ZnWn[1], linewidth = 4, label = r'$Z(\partial_x^2 < 0), W(\partial_x^2 < 0)$', color = 'y', linestyle = '-')
                
            else:
                ax1.plot(d2xZ_ZpWp[0]/1000, d2xZ_ZpWp[1]+offset, linewidth = 4, color = 'g', linestyle = '-')
                ax1.plot(d2xZ_ZpWn[0]/1000, d2xZ_ZpWn[1]+offset, linewidth = 4, color = 'r', linestyle = '-')
                ax1.plot(d2xZ_ZnWp[0]/1000, d2xZ_ZnWp[1]+offset, linewidth = 4, color = 'b', linestyle = '-')
                ax1.plot(d2xZ_ZnWn[0]/1000, d2xZ_ZnWn[1]+offset, linewidth = 4, color = 'y', linestyle = '-')
            if len(lambda_c_list) > 1:
                ax1.text(X[-1]/1000, Z[-1] + offset, r"$\lambda_c=%s$" % lambda_with_unit)
            offset += 3
        ax1.set_xlabel("xs (km)")
        ax1.set_ylabel("Z(m)")
        ax1.legend()
        ax2.plot(X/1000, W, "k-", label = "W")
        ax2.set_xlabel("xs (km)")
        ax2.set_ylabel("W (m)")
        figure.suptitle(r"Segmentation with various $\lambda_c$ values")
        figure.tight_layout(rect=[0, 0, 1, 0.95])
        if args.plot_file is not None:
            plt.savefig(args.plot_file)
        else:
            plt.show()
    
        
    
    
