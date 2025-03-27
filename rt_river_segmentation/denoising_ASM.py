import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pyrscwt import cwt, icwt
import os.path

os.chdir("/Users/sabad-maradona/Documents/KAR/rt-segmentation-rivieres_20241210/rt-segmentation-rivieres/rt_river_segmentation")
from find_best_mother_wavelet import *


def _symetrisation3_(x, y, N):

    # identify if up or downstream flow -> flip the signal to work on downstream
    
    dt = np.diff(x)[0]
    dHdx = np.mean(np.diff(y) / np.diff(x))     
    
    if dHdx>0:    
       y=np.flip(y)
       
    x_sym = x
    y_sym = y

    # gauche du signal  
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
        
            
            
    # droite
    x_sym = np.concatenate((x_sym, x_sym[-1]  + (x_sym[1:])- x_sym[1] ))  
    # plt.plot(x_sym)
    # plt.show()  
      
    V = np.flip(y_sym[0:-1])
    y_sym = np.concatenate((y_sym, V[0] - abs(V - V[0])) )

    # Supprime le morceau en trop
    x_sym = x_sym[1:(N*2+1)*len(x)]
    y_sym = y_sym[1:(N*2+1)*len(x)]     
            
    if dHdx>0:    
       x_sym=x[1]-np.flip(x_sym)+x[-1]
       y_sym=np.flip(y_sym)  
       y=np.flip(y)
       
    
    
    # plt.plot(x_sym/1000, y_sym, color='b' , label = 'Signal Sym')
    # plt.plot(x/1000,y, color='r' , label = 'Signal')

    # plt.xlabel("X (km)")
    # plt.ylabel("Z (m)")

    # plt.legend() 
    # plt.title("Symétrisation signal k" ) 
    # plt.show()  
    
    
    # N = x.size
    # i0 = N_sym * N 
    # iN = (N_sym+1) * N 
    # plt.plot(x_sym[i0:iN]/1000, y_sym[i0:iN], color='b' , label = 'Signal Sym')
    # plt.plot(x/1000,y, color='r' , label = 'Signal')

    # plt.xlabel("X (km)")
    # plt.ylabel("Z (m)")

    # plt.legend() 
    # plt.title("Symétrisation signal k" ) 
    # plt.show()   
    
    return x_sym, y_sym


def hydraulic_filtering_ASM(x, h, N_sym, plot_steps=False):

    dir = -1 # dir uniquement pour la symétrisation 
    
    if (np.mean(np.diff(h) / np.diff(x)))>0:    
        h=np.flip(h) 
        dir = 1

        
    x_sym, H_sym = _symetrisation3_(x, h, N_sym)
    
    # supprime les points de même valeurs si il y en a 
    H_sym = np.delete(H_sym, np.where(np.diff(x_sym)==0))               
    x_sym = np.delete(x_sym, np.where(np.diff(x_sym)==0))

    # Identifie l'ondelette mère a utiliser
    mother, parameter = find_best_mother_wavelet(x, h)
    param = int(parameter)

    # Bornes du domaine signal d'origine avant symétrisation
    N = x.size
    i0 = N_sym * N 
    iN = (N_sym+1) * N 


    dt = np.diff(x)[0]
    dHdx = np.diff(H_sym) / np.diff(x_sym)
        
    # Compute decomposition
    j1 = 40
    waves, period, scales, coi, dj, freqs = cwt(H_sym, dt, dj=None, s0=None, j1=j1, mother=mother, param=param)

    
    inverse_slopes_count = 1 # initialisation du paramètre pour stopper l'itération lorsqu'il n'y a plus de contre-pentes
    index_scale = 0 
    dxfilt = dt
    
    cnt = 1
    vect = np.zeros(x_sym.shape)
    
    Hrec1, freqs = icwt(waves, scales, freqs, dt, dj, mother, param)
    Hrec1 += np.mean(H_sym) # recale le signal reconstruit à la bonne altitude (sinon il est centré autour de zéro)
    
    waves_rec = waves
    
    
    while inverse_slopes_count >=1 :
        
        if x[-1]-x[1]<40000:                
            i_scale=1
        else:
            i_scale=5
            
        
        
        if cnt<i_scale:        
          lambda_coup = scales[index_scale]
        else:
          lambda_coup = scales[i_scale]+cnt*np.mean(np.diff(x))

        cnt += 1
        print(cnt)

        
        slope_lim = 1e-6  # tolerance pente positive à 1e-6 et pas 0 (ajoutée pour le Maroni) à valider       
        dHdx = np.diff(Hrec1) / np.diff(x_sym)
        dHdx[dHdx>slope_lim]=1
        dHdx[dHdx<slope_lim]=0
 
        # pour éviter les problèmes de bords mais ok car dans le domaine du signal symétrisé
        dHdx[0:2]=0
        dHdx[-2:-1]=0
 
        diff_S = np.diff(dHdx)
       
        # 
        # if diff_S[0]==-1:
        #    diff_S[0]=0
        # if diff_S[-1]==1:
        #    diff_S[-1]=0
               
        # Défini le début et la fin des contre-pentes
        indices_in = np.array(np.where(diff_S == 1))+1  # plus car decalage lié a diff
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

        #waves_rec, period, scales, coi, dj, freqs = cwt(Hrec1, dt, dj=None, s0=None, j1=j1, mother=mother, param=param)
        if plot_steps:
           plt.plot(x,Hrec1[i0:iN] , color='b' , label = 'Z rec')         
           plt.show()  



        dHdx_rec = np.diff(Hrec1[i0:iN]) / np.diff(x_sym[i0:iN])
        inverse_slopes = np.array(np.where(dHdx_rec>slope_lim))            
        inverse_slopes_count = inverse_slopes.shape[1]
        
        # if inverse_slopes_count==1:
        #    inverse_slopes_count=0
            
   
    if dir == 1:      
        Hrec1=np.flip(Hrec1)

    i0 = N_sym * (N) 
    iN = (N_sym+1) * (N)     


    #Hrec = Hrec1 - Hrec1[i0] + H_sym[i0]
    Hrec = Hrec1


    return Hrec[i0+1:iN+1] 


