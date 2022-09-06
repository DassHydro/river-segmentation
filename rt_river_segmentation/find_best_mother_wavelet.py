#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:53:48 2022

@author: joao
"""

import numpy as np
import matplotlib.pyplot as plt
from .symetrisation3 import *
from .pyrscwt import cwt, icwt

def find_best_mother_wavelet(X, Z):
    '''
    Choose the best wavelet type (between Morlet, Gaussian and Paul) and parameter for decomposing and 
    recomposing a signal, with the minimal error.
    The variance and the RMSE metrics are calculated as error metrics.

    Parameters
    ----------
    X : array
        abscissa of the givan signal.
    Z : array
        values of the given signal.

    Returns
    -------
    wave_type : string
        best wavelet type for decomposing and recomposing the signal.
    parameter : int
        best wavelet order for decomposing and recomposing the signal.

    '''
    
    # font = {'family' : 'normal',
    #     'weight' : 'bold',
    #     'size'   : 22}
    
    wave = ['morl', 'dog', 'paul'];
    # wave = ['paul']
    
    cnt = 0
    params_arr = np.arange(2, 9, 2)
    
    mat = np.zeros((len(wave)*len(params_arr), 4))
    
    # plt.figure()
    # plt.rc('font', **font)
    # plt.plot(X, Z, label = "Elevation signal Z", linewidth = 4)

    for w in range(0, len(wave)):
        for param in params_arr:
            # x_sym, y_sym = symetrisation3(X, Z, 5)
            # period = np.nanmean(np.diff(X))
            # new_x = np.arange(x_sym[0], x_sym[-1], period)
            # sig = np.interp(new_x, x_sym, y_sym)
            
            x_sym, y_sym = symetrisation3(X, Z, 5)
            period = np.nanmean(np.diff(X))
            # new_x = np.arange(x_sym[0], x_sym[-1], period)
            # sig = np.interp(new_x, x_sym, y_sym)
            
            # CWT decomposition
            if w == 0:
                # mother = pycwt.Morlet(float(param))
                mother = "morlet"
                param = float(param)
            elif w == 1:
                # mother = pycwt.DOG(param)
                mother = "dog"
                param = int(param)
            elif w == 2:
                # mother = pycwt.Paul(param)
                mother = "paul"
                param = int(param)
            
            # mother = pycwt.Morlet(6.)
            # mother = "paul"
            # param = 10
            # dt = period # dt is dx = 10m (interpolated at constant spacing)
            # s0 = 2 * dt  # Starting scale, in this case 2 * dt
            # dj = 1 / 12  # Twelve sub-octaves per octaves
            # J = int(24 / dj)*2  # 24 powers of two with dj sub-octaves
            dt = 10 # dt is dx = 10m (interpolated at constant spacing)
            s0 = None  
            dj = None  
            J = 35  
            # waves, period, scales, coi, dj, freqs = cwt(sig, dt, dj, s0, J, mother, param)     # continuous wavelet decomposition
            waves, period, scales, coi, dj, freqs = cwt(y_sym, dt, dj, s0, J, mother, param)     # continuous wavelet decomposition
            Z_rec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, param)     # signal reconstruction
            Z_rec1 += np.mean(y_sym)
            # Z_rec = Z_rec1[2*len(X):3*len(X)] - Z_rec1[2*len(X)] + Z[0]
            # Z_rec = Z_rec1[2*len(X):3*len(X)] - y_sym[2*len(X)] + sig[0]
            Z_rec = Z_rec1[2*len(X):3*len(X)] - y_sym[2*len(X)] + Z[0]
            
            # plt.figure()
            # plt.plot(X, Z, label = "Z")
            # plt.plot(X, Z_rec, label = "Z_rec, wavelet : %s %i"%(mother, param), linestyle = "--")
            # plt.plot(x_sym, y_sym, "--", label = "y_sym")
            # plt.plot(x_sym, Z_rec1, "--", label = "Z_rec1")
            # plt.legend()
            # # plt.show()
            # plt.savefig("/home/joao/Documents/RT_segmentation/Code/Scripts-Python/Segmentation/plots_comp_signals/ZandZrec_scales" + str(J) + " " + mother + str(param) + ".jpg")
            # # plt.close()
            # plt.close(plt.gcf())
            
            # error metrics calculation (variance and RMSE)
            E = np.sum(np.sqrt(np.square(Z-Z_rec)))
            # rmse = np.nanmean(np.sqrt(np.square(Z-Z_rec)))
            rmse = np.sqrt(np.nanmean((Z-Z_rec)**2))
            
            mat[cnt, :] = [w, param, E, rmse]
            
            cnt += 1
            
            print("mother : {}, param : {}, RMSE : {}".format(wave[w], param, rmse))
    
    # plt.xlabel("xs (m)")
    # plt.ylabel("Z (m)")
    # plt.title("Signal reconstruction after wavelet decomposition: comparison of different wavelets")
    # plt.legend(prop={'size': 8})
    # plt.show()
    
    # plt.rcdefaults()
    
    ind_minE = np.argmin(mat[:, 2])
    ind_minRMSE = np.argmin(mat[:, 3])
    
    wave_type = wave[int(mat[ind_minRMSE, 0])]
    parameter = mat[ind_minRMSE, 1]
    print("mother : {}, param : {}".format(wave_type, parameter))
    
    return wave_type, parameter
            
        
