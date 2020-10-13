# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones(x.shape[0], dtype=int)
    for d in range(1,degree+1):
        poly = np.c_[poly, np.power(x,d)]
    return poly
