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
    # **********************************************************************************************
    # Name:    normalize
    # Purpose: function returns the normalize vector of vec_in
    # Input:   vec_in (1 x n vector) - vector to be normalized
    # Output:  norm_vec_out (1 x n vector) - normalized vector of vec_in
    # Notes:   none
    # **********************************************************************************************
    return vec_in / np.linalg.norm(vec_in)


def normalize_rows(mat_in):
    # **********************************************************************************************
    # Name:    normalize_rows
    # Purpose: function returns the normalized rows of a matrix
    # Input:   vec_in (m x n matrix) - matrix with m row vectors to be normalized
    # Output:  norm_mat_out (m x n matrix) - matrix with m normalized vectors of mat_in
    # Notes:   none
    # **********************************************************************************************

    # calculate the norms of each row in the matrix
    norms = np.linalg.norm(mat_in, axis=1).reshape((-1, 1))

    # normalize each row of the matrix
    norm_mat_out = mat_in / norms

    return norm_mat_out
