#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:58:46 2022

@author: joao
"""

import numpy as np
import pandas as pd
#import scipy.signal as sps
import matplotlib.pyplot as plt
from .symetrisation3 import *
from .find_best_mother_wavelet import *
from .find_regional_peaks import *
from .pyrscwt import cwt, icwt


def segmentation_baseline(X, Z, Zb, lambda_c, wavelet=None, min_length=None, plot=False):
    '''
    Segmentation of a given signal. The signal is first filtered using a continuous wavelet decomposition 
    (the wavelet is chosen automatically), then reconstructed. Its curvature is calculated at every abscissa, 
    and the values are used for the decomposition.

    Parameters
    ----------
    X : array
        array of abscissa of the signal.
    Z : array
        array of water surface elevation values of the signal.
    Zb : array
        array of river bottom elevation values of the signal.
    lambda_c : float
        DESCRIPTION.

    Returns
    -------
    d2xZ_pos : signal [array, array]
        points where the the initial signal [X, Z] has a positive curvature.
    d2xZ_neg : signal [array, array]
        points where the the initial signal [X, Z] has a negative curvature.

    '''
    
    pas = np.nanmean(np.diff(X))
    
    # Filtering the signal in order to calculate the curvature
    vect_freq_coup = lambda_c/1.795195802051312
    # ind_segments_seg1 = np.full((len(vect_freq_coup), len(X)), float("Nan"))
    ind_segments_seg1 = np.full_like(X, float("Nan"))
    cnt = 0
    
    if wavelet is None:
        print("\n" + "-" * 80)
        print("Automatic selection of mother wavelet")
        print("-" * 80)
        wave_type, parameter = find_best_mother_wavelet(X, Z)
        print("-" * 80 + "\n")
    else:
        wave_type, parameter = wavelet
    
    print("\n" + "-" * 80)
    print("Segmentation (baseline)")
    print("-" * 80)

    # for fc in range(0, len(vect_freq_coup)):
    fc = 0
    
    Z_f = np.full_like(X, float("NaN"))
    dxZ_f = np.full_like(X, float("NaN"))
    d2xZ_f = np.full_like(X, float("NaN"))
    
    x_sym, y_sym = symetrisation3(X, Z, 5)
    period = np.nanmean(np.diff(X))
    new_x = np.arange(x_sym[0], x_sym[-1]+period, period)
    sig = np.interp(new_x, x_sym, y_sym)
    
    # CWT decomposition
    if wave_type == "morl":
        # mother = pycwt.Morlet(float(parameter))
        mother = "morlet"
    elif wave_type == "dog":
        # mother = pycwt.DOG(parameter)
        mother = "dog"
        parameter = int(parameter)
    elif wave_type == "paul":
        # mother = pycwt.Paul(parameter)
        mother = "paul"
        parameter = int(parameter)
    
    # mother = pycwt.Morlet(6.)
    # mother = "paul"
    # parameter = 10
    # dt = period # dt is dx = 10m (interpolated at constant spacing)
    # s0 = 2 * dt  # Starting scale, in this case 2 * dt
    # dj = 1 / 12  # Twelve sub-octaves per octaves
    # J = int(24 / dj)  # 24 powers of two with dj sub-octaves
    dt = 10 # dt is dx = 10m (interpolated at constant spacing)
    s0 = None  # Starting scale, in this case 2 * dt
    dj = None  # Twelve sub-octaves per octaves
    J = 35  # 24 powers of two with dj sub-octaves
    waves, period, scales, coi, dj, freqs = cwt(sig, dt, dj, s0, J, mother, parameter)    # continuous wavelet decomposition
    waves[scales < vect_freq_coup] = 0     # filtering of the signal using a wavelength threshold
    
    # unused in the Matlab program
    # print(np.shape(scales))
    # print(np.shape(fftfreqs))
    # C = np.polyfit(scales, fftfreqs, 1)
    # freq_coup_m = C[0]*vect_freq_coup
    
    # reconstruction of the signal
    Z_rec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, parameter)
    Z_rec = Z_rec1[2*len(X):3*len(X)] - Z_rec1[2*len(X)] + Z[0]
    # Z_rec = Z_rec1[2*len(X):3*len(X)] - y_sym[2*len(X)] + sig[0]
    
    # slope and curvature calculation
    Z_f = np.copy(Z_rec)
    dxZ_f[:-1] = np.diff(Z_f)/np.diff(X)     # slope
    d2xZ_f[1:] = np.diff(dxZ_f)/np.diff(X)     # curvature
    
    
    # differentiating points of the initial signal where the curvature is positive, or negative
    M_pos = np.full_like(Z, float("NaN"))
    X_pos = np.full_like(Z, float("NaN"))
    X_pos[d2xZ_f > 0] = X[d2xZ_f > 0]
    M_pos[d2xZ_f > 0] = Z[d2xZ_f > 0]
    M_neg = np.full_like(Z, float("NaN"))
    X_neg = np.full_like(Z, float("NaN"))
    X_neg[d2xZ_f < 0] = X[d2xZ_f < 0]
    M_neg[d2xZ_f < 0] = Z[d2xZ_f < 0]
    
    X_inflexion = []
    Z_inflexion = []
    for i in range(0, len(X_pos)-1):
        if (np.isfinite(X_pos[i]) & np.isnan(X_pos[i+1])):
            x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_pos[i], X_neg[i+1]])
            z_inf = np.interp(x_inf, X, Z)
            X_inflexion.append(x_inf)
            Z_inflexion.append(z_inf)
        elif (np.isfinite(X_neg[i]) & np.isnan(X_neg[i+1])):
            x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_neg[i], X_pos[i+1]])
            z_inf = np.interp(x_inf, X, Z)
            X_inflexion.append(x_inf)
            Z_inflexion.append(z_inf)
    
    X_inflexion = np.asarray(X_inflexion)
    Z_inflexion = np.asarray(Z_inflexion)
    
    X_regular = np.arange(X[0], X[-1], 2000)
    Z_regular = np.interp(X_regular, X, Z)
    
    # determine indices where there are peaks in curvature values
    val_min = np.nanmax(np.abs(d2xZ_f)) * 0.05
    print("val_min=", val_min)
    ind_pics_positifs, ind_pics_negatifs = find_regional_peaks(d2xZ_f, val_min, debug=False)
    
    if False:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 12))
        axes[0].plot(X_neg, M_neg, "r-")
        axes[0].plot(X_pos, M_pos, "g-")
        axes[0].plot(X[ind_pics_positifs], Z[ind_pics_positifs], 'r+')
        axes[0].plot(X[ind_pics_negatifs], Z[ind_pics_negatifs], 'g+')
        axes[0].plot(X_inflexion, Z_inflexion, 'b+')
        axes[1].plot(X, dxZ_f)
        axes[2].plot(X, d2xZ_f)
        axes[2].plot(X[ind_pics_positifs], d2xZ_f[ind_pics_positifs], 'r+')
        axes[2].plot(X[ind_pics_negatifs], d2xZ_f[ind_pics_negatifs], 'g+')
        axes[0].plot(X_inflexion, Z_inflexion, 'b+')
        for index in ind_pics_positifs:
            axes[0].axvline(X[index], color="k", ls="--")
            axes[2].axvline(X[index], color="k", ls="--")
        for index in ind_pics_negatifs:
            axes[0].axvline(X[index], color="gray", ls="--")
            axes[2].axvline(X[index], color="gray", ls="--")
        axes[2].axhline(0.0, color="b", ls="--")
        axes[2].axhline(val_min, color="g", ls="--")
        axes[2].axhline(-val_min, color="g", ls="--")
        
        plt.show()
    
    
    ind_pos_et_neg = np.concatenate((ind_pics_positifs, ind_pics_negatifs))
    indices = np.sort(ind_pos_et_neg)
    ind_sort = indices[((indices > 0) & (indices < len(X)))]
    indices2 = np.concatenate(((0, ), ind_sort, (len(X) - 1, )))
    ind_segments_seg1[:len(indices2)] = indices2
    
    d2xZ_pos = [X_pos, M_pos]
    d2xZ_neg = [X_neg, M_neg]
    
    # prepare plots of segmentation
    ind_nan_seg1 = np.where(np.isnan(ind_segments_seg1))
    seg1_no_nan = ind_segments_seg1[np.where(~np.isnan(ind_segments_seg1))]
    seg1_no_nan = seg1_no_nan.astype("int")
    
    # Reaches S2 (extremums of curvature)
    reaches_X = np.take(X, seg1_no_nan)
    
    #reaches_X = np.concatenate((reaches_X, X_inflexion))
    #reaches_X = np.sort(reaches_X)
    
    # Compute reaches lengths
    reach_lengths = np.diff(reaches_X)
    
    if min_length is not None and len(reach_lengths) > 1:
        if (np.any(reach_lengths < min_length)):
            indices_to_remove = np.ravel(np.argwhere(reach_lengths < min_length))
            if indices_to_remove[0] == 0:
                indices_to_remove[0] = 1
            reaches_X = np.delete(reaches_X, indices_to_remove)


    print("- lambda_c : %f" % lambda_c)
    #print(selected_X)
    print("  - reaches count : %i" % len(reach_lengths))
    print("  - reaches lengths : [%.3f - %.3f]" % (np.min(reach_lengths), np.max(reach_lengths)))
    print("-" * 80 + "\n")
    
    if plot:
        figure = plt.figure(figsize = (15, 10))
        plt.plot(X_pos/1000, M_pos, linewidth = 4, label = r'$Z(\partial_x^2 > 0)$', color = 'g', linestyle = '-')
        plt.plot(X_neg/1000, M_neg, linewidth = 4, label = r'$Z(\partial_x^2 < 0)$', color = 'r', linestyle = '-')
        selected_X = np.take(X, seg1_no_nan)
        selected_Z = np.take(Z, seg1_no_nan)
        selected_Zrec = np.take(Z_rec, seg1_no_nan)
        plt.plot(selected_X/1000, selected_Z, "k.", markersize = 16)
        plt.xlabel("xs (km)")
        plt.ylabel("Z(m)")
        plt.title(r"Segmentation $\lambda_c = $" + str(lambda_c/1000) + " km")
        plt.legend()
        plt.show()
        
        ## Zb SEGMENTATION
        #figure2 = plt.figure(figsize = (15, 10))
        #plt.plot(X/1000, Z, linewidth = 4, color = "r", label = "Inflexion points + curvature maxima")
        #plt.plot(selected_X/1000, selected_Z, "r.", markersize = 12)
        #plt.plot(X_inflexion/1000, Z_inflexion, "r.", markersize = 12)
        #plt.plot(X/1000, Z+5, linewidth = 4, color = "r", label = "Inflexion points only")
        #plt.plot(X_inflexion/1000, Z_inflexion+5, "r.", markersize = 12)
        #plt.plot(X/1000, Z+10, linewidth = 4, color = "r", label = "Curvature maxima only")
        #plt.plot(selected_X/1000, selected_Z+10, "r.", markersize = 12)
        #plt.plot(X/1000, Z+15, linewidth = 4, color = "r", label = "Constant reaches boudaries every 2km")
        #plt.plot(X_regular/1000, Z_regular+15, "r.", markersize = 12)
        #plt.xlim(0, np.max(X)/1000)
        #plt.ylim(60, 145)
        #plt.xlabel("xs (km)")
        #plt.ylabel("Z(m)")
        #plt.title(r"Different segmentations with $\lambda_c = $" + str(lambda_c/1000) + " km")
        #plt.legend()
        #plt.show()
        #plt.close(fig = figure2)
    
    return d2xZ_pos, d2xZ_neg, reaches_X, (wave_type, parameter)



def segmentation_advanced(X, Z, Zb, W, Ah, lambda_c, wavelet_Z=None, wavelet_W=None, add_max_curvature_W=False, 
                          min_length=None, plot=False):
    '''
    Segmentation of a given signal, this time using the descriptor W alongside with Z. The signal is first filtered using a continuous wavelet decomposition 
    (the wavelet is chosen automatically), then reconstructed. Its curvature is calculated at every abscissa, 
    and the values are used for the decomposition.

    Parameters
    ----------
    X : array
        array of abscissa of the signal.
    Z : array
        array of water surface elevation values of the signal.
    Zb : array
        array of river bottom elevation values of the signal.
    W : array
        array of river width values of the signal.
    lambda_c : float
        DESCRIPTION.

    Returns
    -------
    d2xZ_pos : signal [array, array]
        points where the the initial signal [X, Z] has a positive curvature.
    d2xZ_neg : signal [array, array]
        points where the the initial signal [X, Z] has a negative curvature.

    '''
    
    # TODO remove Ah and Zb from arguments
    
    pas = np.nanmean(np.diff(X))
    
    # Filtering the signal in order to calculate the curvature
    vect_freq_coup = lambda_c/1.795195802051312
    # ind_segments_seg1 = np.full((len(vect_freq_coup), len(X)), float("Nan"))
    ind_segments_seg1 = np.full_like(X, float("Nan"))
    cnt = 0
    
    # Z WAVELET TRANSFORM AND RECONSTRUCTION, DERIVATIVES CALCULATION-------------
    
    #print("-----Segmentation of Z-----")
    if wavelet_Z is None:
        print("\n" + "-" * 80)
        print("Automatic selection of mother wavelet for Z")
        print("-" * 80)
        wave_type, parameter = find_best_mother_wavelet(X, Z)
        print("-" * 80 + "\n")
    else:
        wave_type, parameter = wavelet_Z 
    
    # for fc in range(0, len(vect_freq_coup)):
    fc = 0
    
    Z_f = np.full_like(X, float("NaN"))
    dxZ_f = np.full_like(X, float("NaN"))
    d2xZ_f = np.full_like(X, float("NaN"))
    
    x_sym, y_sym = symetrisation3(X, Z, 5)
    period = np.nanmean(np.diff(X))
    new_x = np.arange(x_sym[0], x_sym[-1]+period, period)
    sig = np.interp(new_x, x_sym, y_sym)
    
    # CWT decomposition
    if wave_type == "morl":
        # mother = pycwt.Morlet(float(parameter))
        mother = "morlet"
    elif wave_type == "dog":
        # mother = pycwt.DOG(parameter)
        mother = "dog"
        parameter = int(parameter)
    elif wave_type == "paul":
        # mother = pycwt.Paul(parameter)
        mother = "paul"
        parameter = int(parameter)
    
    # mother = pycwt.Morlet(6.)
    # mother = "paul"
    # parameter = 10
    # dt = period # dt is dx = 10m (interpolated at constant spacing)
    # s0 = 2 * dt  # Starting scale, in this case 2 * dt
    # dj = 1 / 12  # Twelve sub-octaves per octaves
    # J = int(24 / dj)  # 24 powers of two with dj sub-octaves
    dt = 10 # dt is dx = 10m (interpolated at constant spacing)
    s0 = None  # Starting scale, in this case 2 * dt
    dj = None  # Twelve sub-octaves per octaves
    J = 35  # 24 powers of two with dj sub-octaves
    waves, period, scales, coi, dj, freqs = cwt(sig, dt, dj, s0, J, mother, parameter)    # continuous wavelet decomposition
    waves[scales < vect_freq_coup] = 0     # filtering of the signal using a wavelength threshold
    
    # unused in the Matlab program
    # print(np.shape(scales))
    # print(np.shape(fftfreqs))
    # C = np.polyfit(scales, fftfreqs, 1)
    # freq_coup_m = C[0]*vect_freq_coup
    
    # reconstruction of the signal
    Z_rec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, parameter)
    Z_rec = Z_rec1[2*len(X):3*len(X)] - Z_rec1[2*len(X)] + Z[0]
    # Z_rec = Z_rec1[2*len(X):3*len(X)] - y_sym[2*len(X)] + sig[0]
    
    # slope and curvature calculation
    Z_f = np.copy(Z_rec)
    dxZ_f[:-1] = np.diff(Z_f)/np.diff(X)     # slope
    d2xZ_f[1:] = np.diff(dxZ_f)/np.diff(X)     # curvature
    
    # SAME FOR W----------------------------------------------------
    
    #print("-----Segmentation of W-----")
    if wavelet_W is None:
        print("\n" + "-" * 80)
        print("Automatic selection of mother wavelet for W")
        print("-" * 80)
        wave_type_W, parameter_W = find_best_mother_wavelet(X, W)
        print("-" * 80 + "\n")
    else:
        wave_type_W, parameter_W = wavelet_W
    
    # for fc in range(0, len(vect_freq_coup)):
    fc = 0
    
    W_f = np.full_like(X, float("NaN"))
    dxW_f = np.full_like(X, float("NaN"))
    d2xW_f = np.full_like(X, float("NaN"))
    
    x_sym_W, y_sym_W = symetrisation3(X, W, 5)
    period = np.nanmean(np.diff(X))
    new_x_W = np.arange(x_sym_W[0], x_sym_W[-1]+period, period)
    sig_W = np.interp(new_x_W, x_sym_W, y_sym_W)
    
    # CWT decomposition
    if wave_type_W == "morl":
        # mother = pycwt.Morlet(float(parameter))
        mother = "morlet"
    elif wave_type_W == "dog":
        # mother = pycwt.DOG(parameter)
        mother = "dog"
        parameter_W = int(parameter_W)
    elif wave_type_W == "paul":
        # mother = pycwt.Paul(parameter)
        mother = "paul"
        parameter_W = int(parameter_W)
    
    # mother = pycwt.Morlet(6.)
    # mother = "paul"
    # parameter = 10
    # dt = period # dt is dx = 10m (interpolated at constant spacing)
    # s0 = 2 * dt  # Starting scale, in this case 2 * dt
    # dj = 1 / 12  # Twelve sub-octaves per octaves
    # J = int(24 / dj)  # 24 powers of two with dj sub-octaves
    dt = 10 # dt is dx = 10m (interpolated at constant spacing)
    s0 = None  # Starting scale, in this case 2 * dt
    dj = None  # Twelve sub-octaves per octaves
    J = 35  # 24 powers of two with dj sub-octaves
    waves, period, scales, coi, dj, freqs = cwt(sig_W, dt, dj, s0, J, mother, parameter_W)    # continuous wavelet decomposition
    waves[scales < vect_freq_coup] = 0     # filtering of the signal using a wavelength threshold
    
    # unused in the Matlab program
    # print(np.shape(scales))
    # print(np.shape(fftfreqs))
    # C = np.polyfit(scales, fftfreqs, 1)
    # freq_coup_m = C[0]*vect_freq_coup
    
    # reconstruction of the signal
    W_rec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, parameter_W)
    W_rec = W_rec1[2*len(X):3*len(X)] - W_rec1[2*len(X)] + W[0]
    # Z_rec = Z_rec1[2*len(X):3*len(X)] - y_sym[2*len(X)] + sig[0]
    
    # slope and curvature calculation
    W_f = np.copy(W_rec)
    dxW_f[:-1] = np.diff(W_f)/np.diff(X)     # first derivative of W
    d2xW_f[1:] = np.diff(dxW_f)/np.diff(X)     # second derivative of W
    
    # DETERMINING THE REACHES BORDERS --------------------------------------------
    
    # differentiating points of the initial signal where the curvature is positive, or negative
    M_ZpWp = np.full_like(Z, float("NaN"))
    X_ZpWp = np.full_like(Z, float("NaN"))
    X_ZpWp[(d2xZ_f > 0) & (d2xW_f > 0)] = X[(d2xZ_f > 0) & (d2xW_f > 0)]
    M_ZpWp[(d2xZ_f > 0) & (d2xW_f > 0)] = Z[(d2xZ_f > 0) & (d2xW_f > 0)]
    M_ZpWn = np.full_like(Z, float("NaN"))
    X_ZpWn = np.full_like(Z, float("NaN"))
    X_ZpWn[(d2xZ_f > 0) & (d2xW_f < 0)] = X[(d2xZ_f > 0) & (d2xW_f < 0)]
    M_ZpWn[(d2xZ_f > 0) & (d2xW_f < 0)] = Z[(d2xZ_f > 0) & (d2xW_f < 0)]
    M_ZnWp = np.full_like(Z, float("NaN"))
    X_ZnWp = np.full_like(Z, float("NaN"))
    X_ZnWp[(d2xZ_f < 0) & (d2xW_f > 0)] = X[(d2xZ_f < 0) & (d2xW_f > 0)]
    M_ZnWp[(d2xZ_f < 0) & (d2xW_f > 0)] = Z[(d2xZ_f < 0) & (d2xW_f > 0)]
    M_ZnWn = np.full_like(Z, float("NaN"))
    X_ZnWn = np.full_like(Z, float("NaN"))
    X_ZnWn[(d2xZ_f < 0) & (d2xW_f < 0)] = X[(d2xZ_f < 0) & (d2xW_f < 0)]
    M_ZnWn[(d2xZ_f < 0) & (d2xW_f < 0)] = Z[(d2xZ_f < 0) & (d2xW_f < 0)]
    
    X_inflexion = []
    Z_inflexion = []
    for i in range(0, len(X_ZpWp)-1):
        if (np.isfinite(X_ZpWp[i]) & np.isnan(X_ZpWp[i+1])):
            if (d2xZ_f[i]*d2xZ_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZpWp[i], X_ZnWp[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
            elif (d2xW_f[i]*d2xW_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZpWp[i], X_ZpWn[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
    
    for i in range(0, len(X_ZpWn)-1):
        if (np.isfinite(X_ZpWn[i]) & np.isnan(X_ZpWn[i+1])):
            if (d2xZ_f[i]*d2xZ_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZpWn[i], X_ZnWn[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
            elif (d2xW_f[i]*d2xW_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZpWn[i], X_ZpWp[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
    
    for i in range(0, len(X_ZnWp)-1):
        if (np.isfinite(X_ZnWp[i]) & np.isnan(X_ZnWp[i+1])):
            if (d2xZ_f[i]*d2xZ_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZnWp[i], X_ZpWp[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
            elif (d2xW_f[i]*d2xW_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZnWp[i], X_ZnWn[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
    
    for i in range(0, len(X_ZnWn)-1):
        if (np.isfinite(X_ZnWn[i]) & np.isnan(X_ZnWn[i+1])):
            if (d2xZ_f[i]*d2xZ_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZnWn[i], X_ZpWn[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
            elif (d2xW_f[i]*d2xW_f[i+1] < 0):
                x_inf = np.interp(0, [d2xZ_f[i], d2xZ_f[i+1]], [X_ZnWn[i], X_ZnWp[i+1]])
                z_inf = np.interp(x_inf, X, Z)
                X_inflexion.append(x_inf)
                Z_inflexion.append(z_inf)
    
    X_inflexion = np.asarray(X_inflexion)
    Z_inflexion = np.asarray(Z_inflexion)
    ind_sort_infpoints = np.argsort(X_inflexion)
    X_inflexion = X_inflexion[ind_sort_infpoints]
    Z_inflexion = Z_inflexion[ind_sort_infpoints]
    
    X_regular = np.arange(X[0], X[-1], 2000)
    Z_regular = np.interp(X_regular, X, Z)
    
    # determine indices where there are peaks in curvature values
    ind_pics_positifs_Z, ind_pics_negatifs_Z = find_regional_peaks(d2xZ_f, 1e-11)
    ind_pics_positifs_W, ind_pics_negatifs_W = find_regional_peaks(d2xW_f, 1e-11)
    if add_max_curvature_W:
        ind_pos_et_neg = np.concatenate((ind_pics_positifs_Z, ind_pics_negatifs_Z, ind_pics_positifs_W, ind_pics_negatifs_W))
    else:
        ind_pos_et_neg = np.concatenate((ind_pics_positifs_Z, ind_pics_negatifs_Z))
    indices = np.sort(ind_pos_et_neg)
    ind_sort = indices[((indices > 0) & (indices < len(X)))]
    indices2 = np.concatenate(((0, ), ind_sort, (len(X) - 1, )))
    ind_segments_seg1[:len(indices2)] = indices2
    
    d2xZ_ZpWp = [X_ZpWp, M_ZpWp]
    d2xZ_ZpWn = [X_ZpWn, M_ZpWn]
    d2xZ_ZnWp = [X_ZnWp, M_ZnWp]
    d2xZ_ZnWn = [X_ZnWn, M_ZnWn]
    
    # prepare plots of segmentation
    ind_nan_seg1 = np.where(np.isnan(ind_segments_seg1))
    seg1_no_nan = ind_segments_seg1[np.where(~np.isnan(ind_segments_seg1))]
    seg1_no_nan = seg1_no_nan.astype("int")
    reaches_X = np.take(X, seg1_no_nan)
    reach_lengths = np.diff(reaches_X)
    
    if min_length is not None and len(reach_lengths) > 1:
        if (np.any(reach_lengths < min_length)):
            indices_to_remove = np.ravel(np.argwhere(reach_lengths < min_length))
            if indices_to_remove[0] == 0:
                indices_to_remove[0] = 1
            reaches_X = np.delete(reaches_X, indices_to_remove)
    
    print("- lambda_c : %f" % lambda_c)
    #print(selected_X)
    print("  - reaches count : %i" % len(reach_lengths))
    print("  - reaches lenghts : [%.3f - %.3f]" % (np.min(reach_lengths), np.max(reach_lengths)))
    print("")
    
    if plot:
        figure, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(X_ZpWp/1000, M_ZpWp, linewidth = 4, label = r'$Z(\partial_x^2 > 0), W(\partial_x^2 > 0)$', color = 'g', linestyle = '-')
        ax1.plot(X_ZpWn/1000, M_ZpWn, linewidth = 4, label = r'$Z(\partial_x^2 > 0), W(\partial_x^2 < 0)$', color = 'r', linestyle = '-')
        ax1.plot(X_ZnWp/1000, M_ZnWp, linewidth = 4, label = r'$Z(\partial_x^2 < 0), W(\partial_x^2 > 0)$', color = 'b', linestyle = '-')
        ax1.plot(X_ZnWn/1000, M_ZnWn, linewidth = 4, label = r'$Z(\partial_x^2 < 0), W(\partial_x^2 < 0)$', color = 'y', linestyle = '-')
        selected_X = np.take(X, seg1_no_nan)
        selected_Z = np.take(Z, seg1_no_nan)
        selected_Zrec = np.take(Z_rec, seg1_no_nan)
        ax1.plot(selected_X/1000, selected_Z, "k.", markersize = 8)
        ax1.plot(X_inflexion/1000, Z_inflexion, "r.", markersize = 8)
        ax1.set_xlabel("xs (km)")
        ax1.set_ylabel("Z(m)")
        ax1.legend()
        ax2.plot(X/1000, W, "k-", label = "W")
        ax2.set_xlabel("xs (km)")
        ax2.set_ylabel("W (m)")
        figure.suptitle(r"Segmentation $\lambda_c = $" + str(lambda_c/1000) + " km")
        figure.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Plot histogram of reach lengths
        all_X = np.concatenate((X_inflexion, selected_X))
        all_Z = np.concatenate((Z_inflexion, selected_Z))
        sort_all_X = np.argsort(all_X)
        all_X = all_X[sort_all_X]
        all_Z = all_Z[sort_all_X]
        reach_lengths = np.diff(all_X)
        plt.figure()
        plt.hist(reach_lengths)
        plt.xlabel("reach length (m)")
        plt.ylabel("Number of reaches for given length")
        plt.title(r"Distribution of reach lengths after segmentation with $\lambda_c = $" + str(lambda_c) + " m")
        plt.show()
    
    return d2xZ_ZpWp, d2xZ_ZpWn, d2xZ_ZnWp, d2xZ_ZnWn, reaches_X, (wave_type, parameter), (wave_type_W, parameter_W)











