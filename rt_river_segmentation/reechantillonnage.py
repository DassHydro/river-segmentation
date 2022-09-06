#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:21:03 2021

@author: joao
"""

import numpy as np

def reechantillonnage(X, Y, pas):
    '''
    Reinterpolate a signal given a step value.

    Parameters
    ----------
    X : array
        array of abscissa of the signal.
    Y : array
        array of values of the signal.
    pas : int
        step.

    Returns
    -------
    x_d : array
        array of abscissas of the extended and symmetrized signal.
    y_d : array
        array of values of the extended and symmetrized signal.

    '''
    
    x_d = np.arange(X[0], X[-1] + pas, pas)
    # y_d = np.interp(X, Y, x_d)
    y_d = np.interp(x_d, X, Y)
    return x_d, y_d
