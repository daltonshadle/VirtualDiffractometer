# **************************************************************************************************
# Name:    math_functions.py
# Purpose: Module to provides math related functions not found in numpy
# Input:   none
# Output:  Module to provides math related functions not found in numpy
# Notes:   none
# **************************************************************************************************


# ********************************************* Imports ********************************************
# Standard Library
import numpy as np


# *************************************** Function Definitions *************************************
def normalize(vec_in):
    # **************************************************************************************************
    # Name:    normalize.py
    # Purpose: function returns the normalize vector of vec_in
    # Input:   vec_in (1 x n vector) - vector to be normalized
    # Output:  norm_vec_out (1 x n vector) - normalized vector of vec_in
    # Notes:   none
    # **************************************************************************************************
    return vec_in / np.linalg.norm(vec_in)
