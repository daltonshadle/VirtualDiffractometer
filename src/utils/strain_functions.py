# **************************************************************************************************
# Name:    strain_functions.py
# Purpose: Declaration and implementation of strain functions
# Input:   none
# Output:  Function definitions for strain calculations from a diffraction experiment
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
# Standard Library
import numpy as np
# My Library
import utils.math_functions as math_func


# *************************************** Function Definitions *************************************
def strain_rosette(lattice_strain, plane_normals):
    # **********************************************************************************************
    # Name:    strain_rosette
    # Purpose: function that determines the strain vector of a crystal by lattice strains and the
    #          rosette equations (Paper: L. Margulies et al. / Acta Materialia 50 (2002) 1771–1779)
    # Input:   lattice_strain (m x 1) - lattice strain values
    #          plane_normals (m x 3) - plane normal directions to project strain tensor to
    # Output:  strain_vec (6 x 1) - crystal strain vectors for m lattice strain values in the
    #                               form [e11, e22, e33, e23, e13, e12]
    # Notes:   none
    # **********************************************************************************************

    # initialize cosine matrix for plane normals
    cos_matrix = np.array([np.power(plane_normals[:, 0], 2),
                           np.power(plane_normals[:, 1], 2),
                           np.power(plane_normals[:, 2], 2),
                           np.multiply(plane_normals[:, 1], plane_normals[:, 2]),
                           np.multiply(plane_normals[:, 0], plane_normals[:, 2]),
                           np.multiply(plane_normals[:, 0], plane_normals[:, 1])])

    # reshape matrix and vector for mldivide
    cos_matrix = np.transpose(cos_matrix)
    cos_matrix = cos_matrix.reshape((-1, 6))
    lattice_strain = lattice_strain.reshape(-1)

    print(plane_normals)
    print(cos_matrix)

    # calculate strain vector
    strain_vec = math_func.mldivide(cos_matrix, lattice_strain)

    # account for precision of mldivide
    precis = 1e-10
    strain_vec[(np.where(np.abs(strain_vec) < precis))] = 0

    return strain_vec


def strain_vec2tensor(strain_vec):
    # **********************************************************************************************
    # Name:    strain_vec2tensor
    # Purpose: function that transforms a strain vector to a strain tensor
    # Input:   strain_vec (6 x 1) - strain vector in the form [e11, e22, e33, e23, e13, e12]
    # Output:  strain_mat (3 x 3) - strain tensor from strain vector
    # Notes:   none
    # **********************************************************************************************

    # calculate strain tensor
    return np.array([[strain_vec[0],     strain_vec[5] / 2, strain_vec[4] / 2],
                     [strain_vec[5] / 2, strain_vec[1],     strain_vec[3] / 2],
                     [strain_vec[4] / 2, strain_vec[3] / 2, strain_vec[2]]]).reshape((3, 3))


def strain_tensor2vec(strain_mat):
    # **********************************************************************************************
    # Name:    strain_tensor2vec
    # Purpose: function that transforms a strain tensor to a strain vector
    # Input:   strain_mat (3 x 3) - strain tensor
    # Output:  strain_vec (6 x 1) - strain vector in the form [e11, e22, e33, e23, e13, e12]
    # Notes:   none
    # **********************************************************************************************

    # calculate strain tensor
    return np.array([strain_mat[0, 0],
                     strain_mat[1, 1],
                     strain_mat[2, 2],
                     strain_mat[1, 2] * 2,
                     strain_mat[0, 2] * 2,
                     strain_mat[0, 1] * 2]).reshape((6, 1))


def calc_lattice_strain_from_two_theta(init_g_sample, init_two_theta, final_two_theta):
    # **********************************************************************************************
    # Name:    calc_lattice_strain_from_two_theta
    # Purpose: function that calculates the lattice strain values based on the two theta values
    #          at two different load steps
    # Input:   init_g_sample (m x 3) - reciprocal lattice vectors in the sample coord system from m
    #                                  diffraction events
    #          init_two_theta (m x 1) - two theta values of the initial load step from m events
    #          final_two_theta (m x 1) - two theta values of the final load step from m events
    # Output:  lattice_strain (m x 1) - lattice
    # Notes:   Units: two theta values are in degrees
    # **********************************************************************************************

    # normalize the reciprocal lattice vectors
    unit_g_sample = math_func.normalize_rows(init_g_sample)

    # calculate lattice strains, from Bragg's Law and definition of strain. Solve for inter-planar
    # spacing with n * lambda = 2 * d * sin(theta), then solve for strain with (d1 - d0) / d0 where
    # d0 is the initial inter-planar spacing and d1 is the final inter-planer spacing
    lattice_strains = (np.sin(np.deg2rad(init_two_theta / 2))
                       / np.sin(np.deg2rad(final_two_theta / 2)) - 1)

    # calculate strain vector using rosette equations and reciprocal lattice vectors
    strain_vec = strain_rosette(lattice_strains, unit_g_sample)

    return strain_vec



