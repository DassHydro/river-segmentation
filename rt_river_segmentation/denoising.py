import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from .pyrscwt import cwt, icwt

def _symetrisation3_(x, y, N):
    x_sym = x
    y_sym = y
    for i in range(2,N+1):
        if i/2 - np.floor(i/2) == 0:
            V = -(np.flip(y))
            y_sym = np.concatenate((y_sym, V[1:] + np.abs(V[0]) + y_sym[-1]))
            x_sym = np.concatenate((x_sym, x_sym[-1] + x[1:] - x_sym[0]))
        else:
            y_sym = np.concatenate((y_sym, y[1:] + y_sym[-1] - y[0]))
            x_sym = np.concatenate((x_sym, x_sym[-1] + x[1:] - x_sym[0]))
    
    return x_sym, y_sym


def hydraulic_filtering(x, H, x_direction="downstream", mother="paul", param=4, plot_steps=False):
    
    if x_direction == "downstream":
        sign = -1.0
    else:
        sign = 1.0
    
    # Compute slopes
    
    
    # Symmetrization on both sizes
    x_sym, H_sym = _symetrisation3_(x, H, 5)
    N = x.size
    i0 = 2 * N - 2
    iN = 3 * N - 2

    dt = np.diff(x)[0]
    dHdx = np.diff(H_sym[i0:iN]) / np.diff(x_sym[i0:iN])
        
    # Compute decomposition
    j1 = 35
    waves, period, scales, coi, dj, freqs = cwt(H_sym, dt, dj=None, s0=None, j1=j1, mother=mother, param=param)


    ## TEST
    #Hrec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, param)
    #Hrec1 += np.mean(H_sym)
    #plt.plot(x, H, "r-")
    #plt.plot(x, Hrec1[i0:iN] - H_sym[i0] + H[0], "b-")
    #plt.show()
    
    inverse_slopes_count = 1
    index_scale = 0
    dxfilt = dt
    while inverse_slopes_count > 0:

        dxfilt = scales[index_scale] / 2.0
        dfilt = int(np.ceil(dxfilt / dt))
        dxfilt = max(1, dxfilt)
        
        # Recompose signal using pyrscwt.icwt
        Hrec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, param)
        Hrec1 += np.mean(H_sym)

        dHdx = np.diff(Hrec1[i0:iN]) / np.diff(x_sym[i0:iN])
        
        inverse_slopes_count = 0
        
        for i in range(0, N-1):
            
            if dHdx[i] * sign <= -1e-12:
                
                inverse_slopes_count += 1

                for j in range(i-dfilt+1, i+dfilt+2):
                    #print("clear wave: (%i, %i)" % (min_scales[i0+j], i0+j))
                    waves[0:index_scale+1, i0+j] = 0
                   
        if plot_steps:
            S = np.real(waves) != 0
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(S[:, i0:iN], aspect="auto")
            plt.title("Inverse slopes count: %i" % inverse_slopes_count)
            ax2.plot(H_sym, "r-")
            ax2.plot(Hrec1 - H_sym[i0] + H[0], "b-")
            ax3.plot(x*0.001, H, "r-")
            ax3.plot(x*0.001, Hrec1[i0:iN] - H_sym[i0] + H[0], "b-")
            ax3.set_xlim(34, 38)
            ax3.set_ylim(95, 98)
            plt.show()
        else:
            print(" - scale %f (%i) : %i adverse slopes detected" % (scales[index_scale], index_scale, inverse_slopes_count))
        
        index_scale += 1
        
    return Hrec1[i0:iN] - H_sym[i0] + H[0]
