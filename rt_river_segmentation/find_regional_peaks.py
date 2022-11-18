#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:37:19 2021

@author: joao
"""

import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from .symetrisation3 import *

def find_regional_peaks_old(X, val_min):
    '''
    Find peaks in a given signal (OLD version: translation from MATLAB code from A. Montazem). The peaks must have a minimum height of val_min.

    Parameters
    ----------
    X : array
        array of the values of the given signal.
    val_min : float
        minimal height for the peaks.

    Returns
    -------
    ind_pic_positifs : array
        array of indices of the positive peaks.
    ind_pic_negatifs : array
        array of indices of the negative peaks.

    '''
    
    X2 = np.copy(X)
    X2[X>0] = 1
    X2[X<0] = -1
    X3 = np.diff(X2)
    
    ind_no_nan = np.argmin(~(np.isnan(X)))
    if X[[ind_no_nan][-1]] > 0:
        X3[len(X2)-2] = -2
    else:
        X3[len(X2)-2] = 2
    
    ind = np.where(((X3 != 0) & (~(np.isnan(X3)))))
    ind_segments = np.concatenate((np.zeros(1), ind[0]))
    ind_segments = ind_segments.astype("int64")
    
    ind_pic_positifs = []
    ind_pic_negatifs = []
    cnt_pos = 0
    cnt_neg = 0
    
    for ii in range(1, len(ind_segments)):
        X4 = np.full_like(X, float("NaN"))
        X4[ind_segments[ii-1]:ind_segments[ii]] = X[ind_segments[ii-1]:ind_segments[ii]]
        if X3[ind_segments[ii]] < 0:
            u = sps.find_peaks(X4, height = val_min)
            u = u[0]
            if list(u):
                v = np.max(X4[u])
            else:
                v = None
            if v:
                ind_pic_positifs += [u[0]]
                cnt_pos += 1
            elif ((not v) & (np.abs(np.max(X4)) > val_min)):
                vv = np.max(X4)
                ind_pic_positifs += [vv]
                cnt_pos += 1
                
        elif X3[ind_segments[ii]] > 0:
            u = sps.find_peaks(-X4, height = val_min)
            u = u[0]
            if list(u):
                v = np.max(-X4[u])
            else:
                v = None
            if v:
                ind_pic_negatifs += [u[0]]
                cnt_neg += 1
            elif ((not v) & (np.abs(np.max(X4)) > val_min)):
                vv = np.max(X4)
                ind_pic_negatifs += [vv]
                cnt_neg += 1
                
    return ind_pic_positifs, ind_pic_negatifs


def find_regional_peaks(X, val_min, debug=False):
    '''
    Find peaks in a given signal. The peaks must have a minimum height of val_min.

    Parameters
    ----------
    X : array
        array of the values of the given signal.
    val_min : float
        minimal height for the peaks.

    Returns
    -------
    ind_pic_positifs : array
        array of indices of the positive peaks.
    ind_pic_negatifs : array
        array of indices of the negative peaks.

    '''
    
    X2 = np.copy(X)
    X2[X>0] = 1
    X2[X<0] = -1
    X3 = np.diff(X2)
    
    ind_no_nan = np.argmin(~(np.isnan(X)))
    if X[[ind_no_nan][-1]] > 0:
        X3[len(X2)-2] = -2
    else:
        X3[len(X2)-2] = 2
    
    ind = np.where(((X3 != 0) & (~(np.isnan(X3)))))
    ind_segments = np.concatenate((np.zeros(1), ind[0]))
    ind_segments = ind_segments.astype("int64")
    
    ind_pic_positifs = []
    ind_pic_negatifs = []
    cnt_pos = 0
    cnt_neg = 0
    
    for ii in range(1, len(ind_segments)):
        X4 = np.full_like(X, float("NaN"))
        X4[ind_segments[ii-1]:ind_segments[ii]] = X[ind_segments[ii-1]:ind_segments[ii]]
        if X3[ind_segments[ii]] < 0:
            u = sps.find_peaks(X4, height = val_min)
            u = u[0]
            if u.size > 0:
                ind_pic_positifs += list(u)
                cnt_pos += u.size
            if debug:
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.plot(X)
                ax2.plot(X3)
                ax1.plot(X4)
                plt.title("ii=%i" % ii)
                ax1.plot(u, X[u], "+")
                plt.show()
                
                
        elif X3[ind_segments[ii]] > 0:
            u = sps.find_peaks(-X4, height = val_min)
            u = u[0]
            if u.size > 0:
                ind_pic_negatifs += list(u)
                cnt_neg += u.size
            if debug:
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.plot(X)
                ax2.plot(X3)
                ax1.plot(X4)
                plt.title("ii=%i" % ii)
                ax1.plot(u, X[u], "+")
                plt.show()
                
    return ind_pic_positifs, ind_pic_negatifs
