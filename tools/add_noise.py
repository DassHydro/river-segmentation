#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:08:26 2021

@author: joao
"""

import argparse
import numpy as np
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Add noise to a reference data")
    parser.add_argument("ref", type=str, help="path to reference data file.")
    parser.add_argument("out", type=str, help="path to the output data file.")
    parser.add_argument("-var", type=str, default="H", help="Variable name")
    parser.add_argument("-noise-type", dest="noise_type", type=str, choices=["normal"], default="normal", help="Statistical law for adding noise.")
    parser.add_argument("-noise-params", dest="noise_params", type=float, nargs="+", default=[0.0, 0.25], help="Statistical law for adding noise.")
    args = parser.parse_args()
    
    reference_data = pd.read_csv(args.ref, sep=";")
    
    noisy_data = reference_data.copy()
    
    if args.noise_type == "normal":
        
        noisy_data.loc[:, args.var] += np.random.normal(*args.noise_params, size=noisy_data.shape[0])
        print(np.random.normal(*args.noise_params, size=noisy_data.shape[0]))
        
    noisy_data.to_csv(args.out, index=False, sep=";")
