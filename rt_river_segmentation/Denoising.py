#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@authors : Amanda Samine Montazem - Kevin Larnier - Garambois Pierre-Andre

"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import os.path
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from find_best_mother_wavelet import *
from pyrscwt import cwt, icwt


def reechantillonnage(X, H, pas):
    '''
    Reinterpolate a signal given a step value.

    '''
    
    x_d = np.arange(X[0], X[-1] + pas, pas)
    # y_d = np.interp(X, Y, x_d)
    y_d = np.interp(x_d, X, H)
     

    return x_d, y_d



def _symetrisation3_(x, y, N):

    # identify if up or downstream flow -> flip the signal to work on downstream
        
    dt = np.diff(x)[0]
    dHdx = np.mean(np.diff(y) / np.diff(x))   
    
    if dHdx>0:    
       y=np.flip(y)
       
    x_sym = x
    y_sym = y

    # Left  
    for i in range(0,N):
        
        if  i % 2 == 0:
            pass # Even 
            x_sym = np.concatenate((x_sym[0] - x_sym[-1] + x[0:-1], x_sym))        
            V = np.flip(y[0:-1])
            y_sym = np.concatenate((abs(V[-1]-V)+ y_sym[1],y_sym))
           
        else:
            pass # Odd
 
            x_sym = np.concatenate((x_sym[0] - x_sym[-1] + x[0:-1], x_sym))        
            V = y[0:-1]
            y_sym = np.concatenate((V + y_sym[1]-y[-1], y_sym))
        
            
            
    # Right 
    x_sym = np.concatenate((x_sym, x_sym[-1]  + (x_sym[1:])- x_sym[1] ))  
    # plt.plot(x_sym)
    # plt.show()  
      
    V = np.flip(y_sym[0:-1])
    y_sym = np.concatenate((y_sym, V[0] - abs(V - V[0])) )

    # Delete the unnecessary section
    x_sym = x_sym[1:(N*2+1)*len(x)]
    y_sym = y_sym[1:(N*2+1)*len(x)]     
            
    if dHdx>0:    
       x_sym=x[1]-np.flip(x_sym)+x[-1]
       y_sym=np.flip(y_sym)  
       y=np.flip(y)
       
    
    return x_sym, y_sym



def hydraulic_filtering(x, h, N_sym, plot_steps=False):


    dir = -1 # dir uniquement pour la symétrisation 
    
    if (np.mean(np.diff(h) / np.diff(x)))>0:    
        h=np.flip(h) 

        dir = 1

        
    x_sym, H_sym = _symetrisation3_(x, h, N_sym)
    
    # Remove duplicate points if there are any
    H_sym = np.delete(H_sym, np.where(np.diff(x_sym)==0))               
    x_sym = np.delete(x_sym, np.where(np.diff(x_sym)==0))

    # Determine the appropriate mother wavelet to use
    mother, parameter = find_best_mother_wavelet(x, h)
    param = int(parameter)

    # Original signal domain boundaries before symmetrization
    N = x.size
    i0 = N_sym * N 
    iN = (N_sym+1) * N 


    dt = np.diff(x)[0]
    dHdx = np.diff(H_sym) / np.diff(x_sym)
        
    # Compute decomposition

    waves, period, scales, coi, dj, freqs = cwt(H_sym, dt, dj=None, s0=None, j1=40, mother=mother, param=param)
    
    
    inverse_slopes_count = 1 # Set the stopping condition to terminate the iteration when all counter-gradients have been removed
    index_scale = 0 
    dxfilt = dt
    
    cnt = 0
    vect = np.zeros(x_sym.shape)
    
    Hrec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, param)
    Hrec1 += np.mean(H_sym) # Restore the reconstructed signal to its proper vertical offset; otherwise, it is zero-centered
    
    waves_rec = waves
    
    
    while inverse_slopes_count >=1 :
        
        if x[-1]-x[1]<40000:                
            i_scale=1
        else:
            i_scale=3
            
        
        
        if cnt<i_scale:        
          lambda_coup = scales[index_scale]
        else:
          lambda_coup = scales[i_scale]+cnt*np.mean(np.diff(x))

        cnt += 1
        print(cnt)

        
        slope_lim = 1e-5  # Tolerance for positive slope is initialized to 1e-5 and may be changed
        dHdx = np.diff(Hrec1) / np.diff(x_sym)
        dHdx[dHdx>slope_lim]=1
        dHdx[dHdx<slope_lim]=0
 
        # This avoids boundary problems, which is fine since we're operating on the symmetrized version of the signal
        dHdx[0:2]=0
        dHdx[-2:-1]=0
 
        diff_S = np.diff(dHdx)
       
               
        # Determine the boundaries of the counter-slopes
        indices_in = np.array(np.where(diff_S == 1))+1  # +1 due to offset related to diff
        indices_fin = np.array(np.where(diff_S == -1))+1
  
        
        vect = np.zeros(x_sym.shape)  
        for i in range(0, indices_in.shape[1]-1): 
            
            Length = indices_fin[0,i]-indices_in[0,i] # lentgh contre-pente         
            vect[indices_in[0,i]-2*Length:indices_fin[0,i]+2*Length] = lambda_coup
                
 
        for ii in range(0,len(vect)-1):
            waves_rec[scales<vect[ii],ii] = 0
        
        
        
        index_scale += 1            
            
            
        ########################  
        
        # Recompose signal using pyrscwt.icwt
        
        Hrec1, freqs = icwt(waves_rec, scales, freqs, dt, dj, mother, param)
        Hrec1 += np.mean(H_sym)



        dHdx_rec = np.diff(Hrec1[i0:iN]) / np.diff(x_sym[i0:iN])
        inverse_slopes = np.array(np.where(dHdx_rec>slope_lim))            
        inverse_slopes_count = inverse_slopes.shape[1]
        

           
   
    if dir == 1:      
        Hrec1=np.flip(Hrec1)

    i0 = N_sym * (N) 
    iN = (N_sym+1) * (N)     


    Hrec = Hrec1


    return Hrec[i0+1:iN+1] 


def denoising(X, Z, N_sym, dx):
    
    h_erreur = 2
    
    dir = -1
    if (np.mean(np.diff(Z) / np.mean(np.diff(X))))>0:    
        Z=np.flip(Z) 
        X=np.flip(X) 
        dir = 1
        
    dZ = np.diff(Z) 

    while np.any(dZ > h_erreur):
        dZ = np.diff(Z) 
        ind_pos = np.where(dZ>h_erreur)[0]+1
        Z = np.delete(Z, [ind_pos-1, ind_pos])
        X = np.delete(X, [ind_pos-1, ind_pos])
 

 
    if dir == 1:    
        Z = np.flip(Z) 
        X = np.flip(X) 
        
    
    
    Xr, Zr = reechantillonnage(X,Z, dx)
    

        
        ##########################################

 
    if  dir == 1:
        Zr=np.flip(Zr) 
        
 
    # Applying a filter to the signal to identify crests and troughs
    X_sym, Z_sym = _symetrisation3_(Xr, Zr, N_sym)
           

    # Remove points with duplicate values if there are any
    Z_sym = np.delete(Z_sym, np.where(np.diff(X_sym)==0))               
    X_sym = np.delete(X_sym, np.where(np.diff(X_sym)==0))

    # Determine the appropriate mother wavelet to use
    mother, parameter = find_best_mother_wavelet(Xr, Zr)
    param = int(parameter)

    # Original signal domain boundaries before symmetrization
    N = Xr.size
    i0 = N_sym * N 
    iN = (N_sym+1) * N 


    dt = np.diff(Xr)[0]
    dZdx = np.diff(Z_sym) / np.diff(X_sym)
        
    # Compute decomposition
    j1 = 40
    waves, period, scales, coi, dj, freqs = cwt(Z_sym, dt, dj=None, s0=None, j1=40, mother=mother, param=param)
    #waves, period, scales, coi, dj, freqs = cwt(Z_sym, dt, dj=None, s0=None, j1=None, mother=mother, param=param)
  
    waves_rec = waves.copy()

    # Recompose filtered signal
    waves_rec[0:8,:] = 0

    Z_rec, freqs = icwt(waves_rec, scales, freqs, dt, dj, mother, param)
    Z_rec += np.mean(Z_sym)  
    
    S_rec = np.diff(Z_rec) / np.mean(np.diff(X_sym))
    C_rec = np.diff(S_rec) / np.mean(np.diff(X_sym))
   
    ####################
    slope_lim = 0  # Tolerance for positive slope is initialized to 0
    dZdx = np.diff(Z_rec) / np.mean(np.diff(X_sym))
    dZdx[dZdx>slope_lim]=1
    dZdx[dZdx<slope_lim]=0
         
    # This avoids boundary problems, which is fine since we're operating on the symmetrized version of the signal
    dZdx[0:1]=0
    dZdx[-2:-1]=0
         
    diff_S = np.diff(dZdx)
               
                       
    # Determine the boundaries of the counter-slopes
    indices_in = np.array(np.where(diff_S == 1))  # +1 due to offset related to diff
    indices_fin = np.array(np.where(diff_S == -1))+1
          
    DZ = np.array([]) 
    Z_rec1 = Z_rec.copy()
            
    for i in range(3, indices_in.shape[1]-2): # exclude the edges of the symmetrized signal from analysis
                    
        Delta_Z = Z_rec1[indices_fin[0,i]]-Z_rec1[indices_in[0,i]] # counter-slope lentgh         
        DZ = np.append(DZ, Delta_Z)
        print(i)
        print(Delta_Z)
                     
        if (Delta_Z>h_erreur) & (C_rec[indices_in[0,i]]>0):   # BUMP
            Length = indices_fin[0,i]-indices_in[0,i] # counter-slope lentgh  
            # Delete the counter-slope length
            X_sym[indices_in[0,i]+2:indices_fin[0,i]] = np.nan
            Z_rec[indices_in[0,i]+2:indices_fin[0,i]] = np.nan
            print(i)
            
            # BUMP removes the length over which the signal decreases negatively by deltaZ
                    
            j=1
            while ((Z_rec1[indices_fin[0,i]+1]-Z_rec1[indices_fin[0,i]+j])<Delta_Z) & (indices_fin[0,i]+j<len(Z_sym)-1): # Second condition avoids surpassing the boundaries of the signal
                  j=j+1               
            X_sym[indices_fin[0,i]:indices_fin[0,i]+j+1] = np.nan
            Z_rec[indices_fin[0,i]:indices_fin[0,i]+j+1] = np.nan
        
        elif  (Delta_Z>h_erreur) & (C_rec[indices_in[0,i]]<0):   # TROUGHS
            Length = indices_fin[0,i]-indices_in[0,i] # couter-slopes lentgh  
            
            # Delete the counter-slope length
            X_sym[indices_in[0,i]+2:indices_fin[0,i]] = np.nan
            Z_rec[indices_in[0,i]+2:indices_fin[0,i]] = np.nan
            print(i)
                  
            # removes the length over which the signal decreases negatively by deltaZ             
            j=1
            while ((Z_rec1[indices_fin[0,i]+1]-Z_rec1[indices_fin[0,i]+j])<Delta_Z) & (indices_fin[0,i]+j<len(Z_sym)-1): # Second condition avoids surpassing the boundaries of the signal 
                  j=j+1   
                 
            X_sym[indices_fin[0,i]:indices_fin[0,i]-j-1] = np.nan
            Z_rec[indices_fin[0,i]:indices_fin[0,i]-j-1] = np.nan
    

    X_NO1 = X_sym[i0-1:iN-1]             
    Z_NO1 = Z_rec[i0-1:iN-1]             
           
    X_NO = Xr[~np.isnan(X_NO1)]
    Z_NO = Zr[~np.isnan(Z_NO1)] 
    
    Xr,Zr= reechantillonnage(X_NO, Z_NO, dx)    


    if dir == 1:    
        Zr = np.flip(Zr) 

 

    Zfiltered = hydraulic_filtering(Xr, Zr, N_sym, plot_steps=False) 
    


        
            
    return Xr, Zfiltered 

     
 

