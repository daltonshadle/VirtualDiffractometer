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
import warnings


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


def mldivide(A, b):
    # **********************************************************************************************
    # Name:    mldivide
    # Purpose: function solves for x in the equation Ax = b using the MATLAB solution of most zeros
    #          in the solution for under-determined systems
    # Input:   A (m x n matrix) - matrix to left divide
    #          b (m x 1 matrix) - solution matrix
    # Output:  x (n x 1 matrix) - solution for x in the equation Ax = b
    # Notes:   none
    # **********************************************************************************************

    # solve Ax=b for x using least squares method
    b = b.reshape((-1, 1))
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    # check if the system is under-determined, m < n, and throw a warning
    if A.shape[0] < A.shape[1]:
        warn_str = ("Under-determined system in utils.math_functions.mldivide: (m, n) = "
                    + str(A.shape))
        warnings.warn(warn_str)
    return x

