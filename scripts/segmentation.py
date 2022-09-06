#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:08:26 2021

@author: Joao Hemptinne (joao.hemptinne@csgroup.eu)
"""

import argparse
from rt_river_segmentation.segmentation_methods import *
from rt_river_segmentation.load_and_process_data import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Procedure to realize a segmentation of a signal of water surface heights")
    parser.add_argument("data_files", type=str, nargs="+", help="path to main input data files (heights, widths).")
    parser.add_argument("-complementary_data", type=str, default="default", help="path to main input data")
    parser.add_argument("-time-index", dest="time_index", type=int, default=0, help="Index of time in the data")
    parser.add_argument("-lambda", dest="lambda_c", type=float, nargs="+", default=1.0, help="Caracteristic length(s) for the segmentation (km)")
    parser.add_argument("-dx", type=int, default=10, help="Spacing to resample data (m)")
    parser.add_argument("-plot-file", dest="plot_file", type=str, default=None, help="Path to the plot file")
    parser.add_argument("--enable-width-max-curvature-points", dest="enable_width_max_curvature_points",
                        action="store_true", 
                        help="Enable maximum of curvature of W for segments definitions ('advanced' only)")
                        
    args = parser.parse_args()

    # INPUTS
    # Resampling spacing
    pas = args.dx
    # Données Garonne sur 365 jours
    time_index = args.time_index
    # Characteristic length(s)
    if isinstance(args.lambda_c, list):
        lambda_c_list = [lambda_c * 1000.0 for lambda_c in args.lambda_c]
    else:
        lambda_c_list = [args.lambda_c * 1000.0]
    
    
    # PROCESSING
    X, W, M_Z, Zb, H, dxZ, dx2Z, dxZb, Q, Ah, Ph = load_and_process_data(args.data_files, args.complementary_data, pas)
    Z = M_Z[time_index, :]
    W = W[time_index, :]
    Ah = Ah[time_index, :]
    
    if len(args.data_files) == 1:
        
        #--------------------------------------------------------------------------------------------------------------
        # Segmentation "baseline" (using heights only)
        #--------------------------------------------------------------------------------------------------------------
        
        # Compute segmentations for every lambda_c
        wavelet = None
        d2xZ_pos_list = []
        d2xZ_neg_list = []
        for lambda_c in lambda_c_list:
            if wavelet is None:
                d2xZ_pos, d2xZ_neg, wavelet = segmentation_baseline(X, Z, Zb, lambda_c)
            else:
                d2xZ_pos, d2xZ_neg, _ = segmentation_baseline(X, Z, Zb, lambda_c, wavelet)
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
                d2xZ_ZpWp, d2xZ_ZpWn, d2xZ_ZnWp, d2xZ_ZnWn, wavelet_Z, wavelet_W = results
            else:
                results = segmentation_advanced(X, Z, Zb, W, Ah, lambda_c, wavelet_Z, wavelet_W,
                                                add_max_curvature_W=args.enable_width_max_curvature_points)
                d2xZ_ZpWp, d2xZ_ZpWn, d2xZ_ZnWp, d2xZ_ZnWn, _, _ = results
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
    
        
    
    
