#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:05:18 2021

@author: joao
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import scipy.signal as sps
import matplotlib.pyplot as plt
from .reechantillonnage import *


def load_data(main_data_path, complement_data_path = "default"):
    '''
    Open given datasets and extract hydrological and geographical data

    Parameters
    ----------
    main_data_path : string
        Path to the main data file, containing all (or some) necessary hydrological information.
    complement_data_path : string, optional
        Path to the complementary data file, containing curvilign abscissa values. The default is "default".

    Raises
    ------
    NameError
        DESCRIPTION.

    Returns
    -------
    M_Z : array
        array of water surface elevations.
    M_Zb : array
        array of river bottom elevations.
    M_W : array
        array of river widths.
    M_Q : array
        array of discharge values.
    M_Ah : array
        array of flow area values.
    M_Ph : array
        array of wetted perimeter values.

    '''
    
    
    H_data_path = main_data_path[0]
    if len(main_data_path) == 2:
        W_data_path = main_data_path[1]
    else:
        W_data_path = None
    splitted_name = H_data_path.split(".")
    extension = splitted_name[-1]
    if extension.lower() == "csv":
        # M_Z = (pd.read_csv(main_data_path, sheet_name = "Z (m)")).to_numpy() # water surface elevation
        # M_Zb = (pd.read_csv(main_data_path, sheet_name = "Zb (m)")).to_numpy() # river bottom elevation
        # M_W = (pd.read_csv(main_data_path, sheet_name = "W (m)")).to_numpy() # river width
        # M_Q = (pd.read_csv(main_data_path, sheet_name = "Q (cms)")).to_numpy() # river discharge
        # M_Ah = (pd.read_csv(main_data_path, sheet_name = "A (m2)")).to_numpy()  # cross sectional flow area
        # M_Ph = (pd.read_csv(main_data_path, sheet_name = "P (m)")).to_numpy()  # wetted perimeter of total cross section
        
        M_Z = (pd.read_csv(H_data_path, sep = ";")).to_numpy() # water surface elevation
        # M_Zb = (pd.read_csv(complement_data_path, sep = ";")).to_numpy() # water surface elevation
        M_Zb = np.zeros_like(M_Z) # river bottom elevation
        # W_data_path = "/home/joao/Documents/RT_segmentation/Code/Scripts-Python/Segmentation/GaronneUpstream_W_day50.csv"
        # W_data_path = "/home/joao/Documents/RT_segmentation/Code/Scripts-Python/Segmentation/SacramentoDownstream_W_day100.csv"
        # W_data_path = "/home/joao/Documents/RT_segmentation/Code/Scripts-Python/Segmentation/swothr_sacramento_overpass_109_riverobs_nominal_20201105_W_cut.csv"
        if W_data_path is not None:
        #W_data_path = "/home/joao/Documents/RT_segmentation/Code/Scripts-Python/Segmentation/swothr_sacramento_overpass_109_riverobs_truth_20201105_W_cut.csv"
        #W_data_path = "./DATA/swothr_sacramento_overpass_109_riverobs_truth_20201105_W_cut2.csv"
            M_W = (pd.read_csv(W_data_path, sep = ";")).to_numpy() # river width
        else:
            M_W = np.ones_like(M_Z) * np.nan
        M_Q = np.zeros_like(M_Z) # river discharge
        M_Ah = np.zeros_like(M_Z)  # cross sectional flow area
        # Ah_data_path = "/home/joao/Documents/RT_segmentation/Code/Scripts-Python/Segmentation/GaronneUpstream_A_day50.csv"
        # M_Ah = (pd.read_csv(Ah_data_path, sep = ";")).to_numpy()  # cross sectional flow area
        M_Ph = np.zeros_like(M_Z)  # wetted perimeter of total cross section
        
        return M_Z, M_Zb, M_W, M_Q, M_Ah, M_Ph
    
    elif (extension.lower() == 'nc' or extension.lower() == 'nc4' and complement_data_path != "default"):
        # Open main data to extract hydrological information
        main_data = nc.Dataset(main_data_path)
        main_data_node = main_data['nodes']
        print(main_data_node['reach_id'][:])
        # reach_to_pick_id = main_data_node['reach_id'][0]
        # nodes_to_pick = np.where((main_data_node['reach_id'][:] == reach_to_pick_id))[0]
        # main_data_node_id = main_data_node['node_id'][:]
        # wse_node = main_data_node['wse'][:]
        # width_node = main_data_node['width'][:]
        # wse_node = main_data_node['wse'][nodes_to_pick]
        # width_node = main_data_node['width'][nodes_to_pick]
        # main_data_node_id = main_data_node['node_id'][nodes_to_pick]
        # M_Z_base = np.mean(wse_node, axis = 1)
        # M_W = np.mean(width_node, axis = 1)
        # M_Z_base = wse_node[wse_node.mask == 0]
        # M_W_base = width_node[wse_node.mask == 0]
        reach_ind = 0
        check = 0
        while check == 0:
            print("--------------------------------")
            print("Iteration number : " + str(reach_ind))
            reach_to_pick_id = main_data_node['reach_id'][reach_ind]
            print(reach_to_pick_id)
            nodes_to_pick = np.where((main_data_node['reach_id'][:] == reach_to_pick_id))[0]
            wse_node = main_data_node['wse'][nodes_to_pick]
            # wse_node = wse_node_mask[wse_node_mask.mask == 0]
            width_node = main_data_node['width'][nodes_to_pick]
            # width_node = width_node_mask[wse_node.mask == 0]
            main_data_node_id = main_data_node['node_id'][nodes_to_pick]
            # M_Z_base = wse_node[wse_node.mask == 0]
            M_Z_base_full = wse_node[wse_node.mask == 0]
            # M_W_base = width_node[wse_node.mask == 0]
            M_W_base_full = width_node[wse_node.mask == 0]
            if len(wse_node[wse_node.mask == 0]) != 0:
                len_p = 10
                p = 1
                count = 1
                reaches_to_pick_ids = [reach_to_pick_id]
                while (len_p != 0 and count <= 5):
                    print(p)
                    print(reach_ind+p)
                    reach_to_pick_id_p = main_data_node['reach_id'][reach_ind + p]
                    print(reach_to_pick_id_p)
                    if reach_to_pick_id_p == main_data_node['reach_id'][reach_ind + p - 1]:
                        p += 1
                        continue
                    nodes_to_pick_p = np.where((main_data_node['reach_id'][:] == reach_to_pick_id_p))[0]
                    wse_node_p = main_data_node['wse'][nodes_to_pick_p]
                    width_node_p = main_data_node['width'][nodes_to_pick_p]
                    main_data_node_id_p = main_data_node['node_id'][nodes_to_pick_p]
                    len_p = len(wse_node_p[wse_node_p.mask == 0])
                    print(M_Z_base_full)
                    print(np.shape(M_Z_base_full))                    
                    print(wse_node_p[wse_node_p.mask == 0])
                    print(np.shape(wse_node_p[wse_node_p.mask == 0]))
                    
                    if np.size(np.shape(M_Z_base_full)) > 1:
                        M_Z_base_full = M_Z_base_full[0]
                    
                    if np.size(np.shape(M_W_base_full)) > 1:
                        M_W_base_full = M_W_base_full[0]
                    
                    if np.size(np.shape(wse_node_p[wse_node_p.mask == 0])) > 1:
                        wse_nomask_p = (wse_node_p[wse_node_p.mask == 0])[0]
                    else:
                        wse_nomask_p = wse_node_p[wse_node_p.mask == 0]
                        
                    if np.size(np.shape(width_node_p[wse_node_p.mask == 0])) > 1:
                        width_nomask_p = (width_node_p[wse_node_p.mask == 0])[0]
                    else:
                        width_nomask_p = width_node_p[wse_node_p.mask == 0]
                    
                    M_Z_base_full = np.concatenate((M_Z_base_full, wse_nomask_p), axis = 0)
                    M_W_base_full = np.concatenate((M_W_base_full, width_nomask_p), axis = 0)
                    reaches_to_pick_ids += [reach_to_pick_id_p]
                    main_data_node_id = np.concatenate((main_data_node_id, main_data_node_id_p), axis = 0)
                    wse_node = np.concatenate((wse_node, wse_node_p), axis = 0)
                    # wse_node = np.concatenate((wse_node[wse_node.mask == 0], wse_node_p[wse_node_p.mask == 0]), axis = 0)
                    print(np.shape(M_Z_base_full))
                
                    if count == 5:
                        print("Check !")
                        check = 1
                    
                    p += 1
                    count += 1
            reach_ind += 1
        M_Z_base = M_Z_base_full
        print(M_Z_base)
        M_W_base = M_W_base_full
        
        print(np.asarray(reaches_to_pick_ids))
            
        # while check == 0:
        #     print("--------------------------------")
        #     print("Iteration number : " + str(reach_ind))
        #     reach_to_pick_id = main_data_node['reach_id'][reach_ind]
        #     nodes_to_pick = np.where((main_data_node['reach_id'][:] == reach_to_pick_id))[0]
        #     wse_node = main_data_node['wse'][nodes_to_pick]
        #     width_node = main_data_node['width'][nodes_to_pick]
        #     main_data_node_id = main_data_node['node_id'][nodes_to_pick]
        #     M_Z_base = wse_node[wse_node.mask == 0]
        #     M_W_base = width_node[wse_node.mask == 0]
        #     if len(wse_node[wse_node.mask == 0]) != 0:
        #         check = 1
        #     reach_ind += 1
        
        # Open complementary data to extract curvilign abscissa
        complement_data = nc.Dataset(complement_data_path)
        complement_nodes = complement_data['nodes']
        # nodes_to_pick = np.where((complement_nodes['reach_id'][:] == reach_to_pick_id))[0]
        nodes_to_pick_mask = np.where((complement_nodes['reach_id'][:] == reaches_to_pick_ids[0]))[0]
        # nodes_to_pick = nodes_to_pick_mask[wse_node.mask == 0][0]
        nodes_to_pick = nodes_to_pick_mask[3:]
        print(np.shape(nodes_to_pick))
        print(np.shape(nodes_to_pick_mask))
        # print(reaches_to_pick_ids)
        for i in reaches_to_pick_ids[1:]:
            nodes_to_pick_p = np.where((complement_nodes['reach_id'][:] == i))[0]
            # nodes_to_pick = np.concatenate((nodes_to_pick, nodes_to_pick_p[nodes_to_pick_p.mask == 0]), axis = 0)
            print(nodes_to_pick)
            print(nodes_to_pick_p)
            nodes_to_pick = np.concatenate((nodes_to_pick, nodes_to_pick_p), axis = 0)
        
        print(np.shape(nodes_to_pick))    
        # complement_node_id = complement_nodes['node_id'][:]
        # print(nodes_to_pick)
        # print([wse_node.mask == 0])
        # print(nodes_to_pick[wse_node.mask == 0])
        ind = nodes_to_pick[wse_node.mask == 0]
        print(ind[0])
        complement_node_id = complement_nodes['node_id'][ind[0]]
        print(np.shape(ind))
        print(np.shape(M_Z_base))
        print(np.shape(wse_node))
        print(np.shape(wse_node[wse_node.mask == 0]))
        print(M_Z_base)
        print(wse_node[wse_node.mask == 0])
        # indices = np.isin(complement_node_id, main_data_node_id)
        indices = np.where(np.isin(complement_node_id, main_data_node_id))
        selected_x = complement_nodes['x'][indices]
        selected_y = complement_nodes['y'][indices]
        selected_xs = complement_nodes['dist_out'][indices]
        print(selected_xs)
        print(np.shape(selected_xs))
        M_Z = np.zeros((len(M_Z_base), 2))
        M_Z[:, 0] = np.flip(selected_xs)
        M_Z[:, 1] = np.flip(M_Z_base)
        M_W = np.zeros((len(M_W_base), 2))
        M_W[:, 0] = np.flip(selected_xs)
        M_W[:, 1] = np.flip(M_W_base)
        M_Zb = np.zeros_like(M_Z)
        M_Q = np.zeros_like(M_Z)
        M_Ah = np.zeros_like(M_Z)
        M_Ph = np.zeros_like(M_Z)
        
        # return M_Z.T, M_Zb.T, M_W.T, M_Q.T, M_Ah.T, M_Ph.T
        return M_Z, M_Zb, M_W, M_Q, M_Ah, M_Ph
        
    elif (extension.lower() == 'xls' or extension.lower() == 'xlsx'):
        M_Z = (pd.read_excel(main_data_path, sheet_name = "Z (m)")).to_numpy() # water surface elevation
        M_Zb = (pd.read_excel(main_data_path, sheet_name = "Zb (m)")).to_numpy() # river bottom elevation
        M_W = (pd.read_excel(main_data_path, sheet_name = "W (m)")).to_numpy() # river width
        M_Q = (pd.read_excel(main_data_path, sheet_name = "Q (cms)")).to_numpy() # river discharge
        M_Ah = (pd.read_excel(main_data_path, sheet_name = "A (m2)")).to_numpy()  # cross sectional flow area
        M_Ph = (pd.read_excel(main_data_path, sheet_name = "P (m)")).to_numpy()  # wetted perimeter of total cross section
        
        return M_Z, M_Zb, M_W, M_Q, M_Ah, M_Ph
    
    else:
        try:
            raise NameError(extension)
        except NameError:
            print("unloadable file type, please give another file with type CSV, NetCDF4 or Excel tab.")
            raise
        


def load_and_process_data(main_data_path, complementary_data_path, pas):
    '''
    Open given datasets. If a step != 0 is given, reinterpolate the datas opened. 

    Parameters
    ----------
    pas : int
        step.

    Returns
    -------
    X_final : array
        final array of abscissas.
    W_final : array
        final array of widths.
    Z_final : array
        final array of water surface elevations.
    Zb_final : array
        final array of river bottom elevations.
    H_final : array
        final array of river depth.
    dxZ_final : array
        final array of water surface slopes.
    dx2Z_final : array
        final array of water surface curvatures.
    dxZb_final : array
        final array of river bottom slopes.
    Q_final : array
        final array of river discharges.
    Ah_final : array
        final array of cross-sectional flow areas.
    Ph_final : array
        final array of wetted perimeter of total cross-section.

    '''
    # nom_dossier='/home/joao/Documents/RT_segmentation/Code/Scripts-Kevin/Data/'
    # nom_fichier='garonne_up_2010'
    # year=2010
    
    # M_Z = (pd.read_excel(nom_dossier+nom_fichier+".xlsx", sheet_name = "Z (m)")).to_numpy() # water surface elevation
    # M_Zb = (pd.read_excel(nom_dossier+nom_fichier+".xlsx", sheet_name = "Zb (m)")).to_numpy() # river bottom elevation
    # M_W = (pd.read_excel(nom_dossier+nom_fichier+".xlsx", sheet_name = "W (m)")).to_numpy() # river width
    # M_Q = (pd.read_excel(nom_dossier+nom_fichier+".xlsx", sheet_name = "Q (cms)")).to_numpy() # river discharge
    # M_Ah = (pd.read_excel(nom_dossier+nom_fichier+".xlsx", sheet_name = "A (m2)")).to_numpy()  # cross sectional flow area
    # M_Ph = (pd.read_excel(nom_dossier+nom_fichier+".xlsx", sheet_name = "P (m)")).to_numpy()  # wetted perimeter of total cross section
    
    M_Z, M_Zb, M_W, M_Q, M_Ah, M_Ph = load_data(main_data_path, complementary_data_path)
    # M_Z[:, 1] = M_Z[:, 1] + np.random.normal(0, 0.25, len(M_Z[:, 1]))
    
    if pas == 0:
        X = M_Z[-1:1:-1, 0]
        Z = M_Z[-1:1:-1, 1:]
        Zb = M_Zb[-1:1:-1, 1]
        W = M_W[-1:1:-1, 1:]
        Q = M_Q[-1:1:-1, 1:]
        Ah = M_Ah[-1:1:-1, 1:]
        Ph = M_Ph[-1:1:-1, 1:]
        H = Z - np.tile(Zb, (365, 1))
        
    else:
        x = M_Z[-1:1:-1, 0]
        z = M_Z[-1:1:-1, 1:]
        zb = M_Zb[-1:1:-1, 1]
        w = M_W[-1:1:-1, 1:]
        q = M_Q[-1:1:-1, 1:]
        ah = M_Ah[-1:1:-1, 1:]
        ph = M_Ph[-1:1:-1, 1:]
        
        # reinterpolation
        xd, Zb = reechantillonnage(x, zb, pas)
        Z = []
        Q = []
        Ah = []
        Ph = []
        W = []
        # for jour in range(0, 365):
        for jour in range(0, np.shape(z)[1]):
            xd, yy = reechantillonnage(x, z[:, jour], pas)  #sens de z ? z.T ?
            Z += [yy]
            xd, yy = reechantillonnage(x, q[:, jour], pas)  #sens de z ? z.T ?
            Q += [yy]
            xd, yy = reechantillonnage(x, ah[:, jour], pas)  #sens de z ? z.T ?
            Ah += [yy]
            xd, yy = reechantillonnage(x, ph[:, jour], pas)  #sens de z ? z.T ?
            Ph += [yy]
            X, yy = reechantillonnage(x, w[:, jour], pas)  #sens de z ? z.T ?
            W += [yy]
        H = Z - np.tile(Zb, (np.shape(z)[1], 1))
    
    # water surface slope calculation
    dxZ = np.full((np.shape(Z)[0], np.shape(Z)[1]), float("NaN"))
    dxZ[:, :-1] = np.diff(Z)/np.tile(np.diff(X),(np.shape(Z)[0], 1))
    # water surface curvature calculation
    dx2Z = np.full((np.shape(Z)[0], np.shape(Z)[1]), float("NaN"))
    dx2Z[:, 1:] = np.diff(dxZ)/np.tile(np.diff(X),(np.shape(Z)[0], 1))
    # river bottom slope calculation
    dxZb = np.full_like(Zb, float("NaN"))
    dxZb[:-1] = np.diff(Zb)/np.diff(X)
    
    # deleting NaN values in the arrays
    ind_no_nan = np.where(~np.isnan(dx2Z[:,1]))
    ind_nan = np.where(np.isnan(dx2Z[:,1]))
    
    dxZ_final = np.delete(dxZ, ind_nan[0], axis = 0)
    dx2Z_final = np.delete(dx2Z, ind_nan[0], axis = 0)
    Zb_final = np.delete(Zb, ind_nan[0], axis = 0)
    Z_final = np.delete(Z, ind_nan[0], axis = 0)
    X_final = np.delete(X, ind_nan[0], axis = 0)
    W_final = np.delete(W, ind_nan[0], axis = 0)
    dxZb_final = np.delete(dxZb, ind_nan[0], axis = 0)
    H_final = np.delete(H, ind_nan[0], axis = 0)
    Q_final = np.delete(Q, ind_nan[0], axis = 0)
    Ah_final = np.delete(Ah, ind_nan[0], axis = 0)
    Ph_final = np.delete(Ph, ind_nan[0], axis = 0)
    
    
    return X_final, W_final, Z_final, Zb_final, H_final, dxZ_final, dx2Z_final, dxZb_final, Q_final, Ah_final, Ph_final





