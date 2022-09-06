#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:12:59 2021

@author: joao
"""

import pywt
import pycwt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def symetrisation3(x, y, N):
    '''
    Extend a signal (x, y) with its own values.

    Parameters
    ----------
    x : array
        array of abscissa of the signal.
    y : array
        array of values of the signal.
    N : int
        number of times the signal is extended.

    Returns
    -------
    x_sym : array
        array of abscissas of the extended and symmetrized signal.
    y_sym : array
        array of values of the extended and symmetrized signal.

    '''
    
    x_sym = x
    y_sym = y
    for i in range(2,N+1):
        if i/2 - np.floor(i/2) == 0:
            V = -(np.flip(y))
            y_sym = np.concatenate((y_sym, V + np.abs(V[0]) + y_sym[-1]))
            x_sym = np.concatenate((x_sym, x_sym[-1] + x - x_sym[0]))
        else:
            y_sym = np.concatenate((y_sym, y + y_sym[-1] - y[0]))
            x_sym = np.concatenate((x_sym, x_sym[-1] + x - x_sym[0]))
    
    return x_sym, y_sym
