# **************************************************************************************************
# Name:    virtual_diffractometer_functions.py
# Purpose: Declaration and implementation of virtual diffractometer functions
# Input:   none
# Output:  Function definitions for virtual diffractometer experiment
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
# Standard Library
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as plot_ani
import scipy.constants as sciconst
# My Library


# *************************************** Function Definitions *************************************
def djshadle_sing_crystal_rot_diff_exp(labsource, grain, hkl_list, omegabounds):
    # **********************************************************************************************
    # Name:    djshadle_sing_crystal_rot_diff_exp
    # Purpose: function that simulates a rotating virtual diffraction experiment on a single crystal
    # Input:   labsource (object) - contains energy and incoming x-ray direction
    #          grain (object) - contains grain information
    #          hkl_list (m x 3) - contains m hkl plane vectors to interrogate for diffraction event
    #          omegabounds (tuple) - contains omega rotation bounds in tuple [low, high, stepSize]
    #                                in degrees
    # Output:  Function outputs a tuple in the form [two_theta, eta, k_out_lab, total_omega]
    #          two_theta (n x 1 matrix) - matrix of two_theta angles for n diffraction events
    #          eta (n x 1 matrix) - matrix of eta angles for n events
    #          k_out_lab (n x 3 matrix) - matrix of outgoing scattering vectors for n events (in lab
    #                                     coord system)
    #          omega (n x 1 matrix) - omega values for n diffraction events
    # Notes:   Units: Angstroms (unit cell and incoming wavelength) and degrees (unit cell and omega
    #          bounds)
    # **********************************************************************************************

    # reciprocal-lattice vectors in crystal and sample coord system (using 2pi convention)
    g_sample = (grain.quat2rotmat() * grain.unitCell.get_reciprocal_lattice_vectors()
                * np.transpose(hkl_list))

    # calc magnitude of reciprocal-lattice vectors, reshape to (m x 1 matrix) where m = # of hkl
    mag_g_sample = np.linalg.norm(g_sample, axis=0)
    mag_g_sample = np.reshape(mag_g_sample, (1, -1))

    # solve for omega (w) using Equ 18 from Pagan, Miller (Connecting heterogeneous single slip to
    # diff peak evolution), using Asin(x) + Bcos(x) = C -> x = arcsin(C / sqrt(A^2 + B^2) + beta
    top = (labsource.kev_2_angstroms() * mag_g_sample ** 2) / (4 * sciconst.pi)
    bottom = np.sqrt(np.power(-g_sample[0, :], 2) + np.power(g_sample[2, :], 2))
    arcsin_inside = np.abs(top / bottom)

    # remove undefined values of arcsin_inside
    bad_arcsin_indices = np.where(arcsin_inside > 1)
    arcsin_inside = np.delete(arcsin_inside, bad_arcsin_indices)
    top = np.delete(top, bad_arcsin_indices)
    bottom = np.delete(bottom, bad_arcsin_indices)
    g_sample = np.delete(g_sample, bad_arcsin_indices, axis=1)

    # calc omega values, stack corresponding values, omega is in radians
    beta = np.abs(np.arcsin(g_sample[2, :] / bottom))
    omega_part1 = -1 * (np.arcsin(arcsin_inside) - beta)
    omega_part2 = sciconst.pi - omega_part1
    omega_part3 = -1 * (np.arcsin(arcsin_inside) + beta)
    omega_part4 = - sciconst.pi - omega_part3

    omega = np.hstack((omega_part1, omega_part2, omega_part3, omega_part4))

    # combine omega with its corresponding reciprocal-lattice vectors (4 x n where n = # of
    # diffraction events)
    # row_1 = omega values
    # row_2 = g_sample_x
    # row_3 = g_sample_y
    # row_4 = g_sample_z
    omega_recip = np.vstack((omega, np.hstack((g_sample, g_sample, g_sample, g_sample))))

    # complete bounds control for omega bounds, keep omega
    bounds = (np.logical_and(np.deg2rad(omegabounds[0]) < omega_recip[0, :],
                             omega_recip[0, :] < np.deg2rad(omegabounds[1])))
    bounds = np.squeeze(np.asarray(bounds))
    omega_recip = omega_recip[:, bounds]

    # build reciprocal lattice vector list in the lab coord system
    sin_omega = np.sin(omega_recip[0, :])
    cos_omega = np.cos(omega_recip[0, :])
    g_lab = np.empty((3, 0), float)

    for y in range(0, np.shape(omega_recip)[1]):
        rotmat_L2C = np.matrix([[cos_omega[0, y], 0, sin_omega[0, y]],
                                [0, 1, 0],
                                [-sin_omega[0, y], 0, cos_omega[0, y]]])
        temp_g_lab = rotmat_L2C * omega_recip[1:4, y]
        g_lab = np.append(g_lab, temp_g_lab, axis=1)

    # build outgoing wave vector list in the lab coord system
    k_in_mat = np.ones(np.shape(g_lab))
    for i in range(0, 3):
        k_in_mat[i, :] = k_in_mat[i, :] * labsource.k_in_lab[i]

    k_out_lab = k_in_mat + g_lab
    # print(k_in_mat, "\n", g_lab)

    # get rid of bad solutions
    precis = 1E-10
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    mag_k_out = np.linalg.norm(k_out_lab, axis=0)

    index = np.where(np.abs((mag_k_out - mag_k_in) / mag_k_in) < precis)
    total_omega = np.squeeze(np.asarray(np.rad2deg(omega_recip[0, index])))
    g_sample = omega_recip[1:4, index]
    g_lab = g_lab[:, index]
    k_out_lab = k_out_lab[:, index]
    k_out_lab = k_out_lab[:, 0]

    # determine two theta and eta
    mag_g = np.linalg.norm(g_sample, axis=0)
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    two_theta = np.rad2deg(2 * np.arcsin(np.divide(mag_g, (2 * mag_k_in))))
    eta = np.rad2deg(np.arctan2(k_out_lab[1, :], k_out_lab[0, :]))

    return [two_theta, eta, k_out_lab, total_omega]


def calc_g_sample(grain, hkl_list):
    # **********************************************************************************************
    # Name:    calc_g_sample
    # Purpose: function calculates the reciprocal lattice vectors for a single crystal
    # Input:   grain (object) - contains grain information
    #          hkl_list (m x 3) - contains m hkl plane vectors to interrogate for diffraction event
    # Output:  g_sample (3 x m) - reciprocal lattice vectors for m hkl planes
    # Notes:   none
    # **********************************************************************************************
    return (grain.quat2rotmat() * grain.reciprocal_strain() *
            grain.unitCell.get_reciprocal_lattice_vectors() * np.transpose(hkl_list))


def calc_omega_rot_diff_exp(g_sample, k_in_lab):
    # **********************************************************************************************
    # Name:    calc_omega_rot_diff_exp
    # Purpose: function that calculates omega values given reciprocal lattice vectors and wavelength
    # Input:   g_sample (3 x m matrix) - contains reciprocal lattice vectors in the sample coord
    #                                    system to calc omega values for
    #          k_in_lab (1 x 3 vector) - vector of the incoming x-ray in the lab coord system
    # Output:  Function outputs a tuple in the form [omega, g_sample]
    #          omega (n x 1 matrix) - omega values for n diffraction events in degrees
    #          g_sample (n x 3 matrix) - reciprocal lattice vectors for n diffraction events
    #          g_sample_index (n x 1 matrix) - gives indices of g_sample used in n events
    # Notes:   omega and g_sample are indexed such that each row index in omega corresponds to the
    #          same row index in g_sample (ie each omega value is pair with a g_sample vector)
    # **********************************************************************************************

    # initialize g_sample_index, ranging across the values in g_sample
    g_sample_index = np.array(range(g_sample.shape[1]))

    # split into x, y, z in the sample coord system
    g_sample_x = np.transpose(g_sample[0, :])
    g_sample_y = np.transpose(g_sample[1, :])
    g_sample_z = np.transpose(g_sample[2, :])

    # set up quadratic formula for solving omega values: qf = (-b +- sqrt(b^2 - 4*a*c)) / (2*a)
    Y = ((np.power(g_sample_x, 2) + np.power(g_sample_y, 2) + np.power(g_sample_z, 2))
         / (2 * np.linalg.norm(k_in_lab)))
    a = np.power(g_sample_x, 2) + np.power(g_sample_z, 2)
    b = 2 * np.multiply(Y, g_sample_x)
    c = np.power(Y, 2) - np.power(g_sample_z, 2)
    rad = np.sqrt((np.power(b, 2) - 4 * np.multiply(a, c)))

    # find solution indices for quadratic formula (index1 = 1 solution & index2 = 2 solutions)
    precis = 1E-6
    index1 = np.where(np.abs(rad < precis))[0]
    index2 = np.where(np.abs(rad >= precis))[0]

    # calculate solution of quadratic formula and find omega values (in degrees)
    plus_solution = np.divide((-np.take(b, index2) + np.take(rad, index2)),
                              (2 * np.take(a, index2)))
    minus_solution = np.divide((-np.take(b, index2) - np.take(rad, index2)),
                               (2 * np.take(a, index2)))
    zero_solution = np.divide(-np.take(b, index1), (2 * np.take(a, index1)))

    plus_omega = np.array(np.rad2deg(np.real(np.arcsin(plus_solution))))
    minus_omega = np.array(np.rad2deg(np.real(np.arcsin(minus_solution))))
    zero_omega = np.array(np.rad2deg(np.real(np.arcsin(zero_solution))))

    # gather all omega solutions and reciprocal lattice vectors in the sample frame, the 180 factor
    # is for finding all possible omega values
    total_omega = np.hstack((plus_omega, np.add(-plus_omega, 180), minus_omega,
                            np.add(-minus_omega, -180), zero_omega))
    g_sample_temp = np.hstack((g_sample[:, index2], g_sample[:, index2],
                              g_sample[:, index2], g_sample[:, index2], g_sample[:, index1]))
    g_sample_index_temp = np.hstack((g_sample_index[index2],
                                     2 * g_sample.shape[1] + (g_sample_index[index2] + 1),
                                     3 * g_sample.shape[1] + (g_sample_index[index2] + 1),
                                     4 * g_sample.shape[1] + (g_sample_index[index2] + 1),
                                     5 * g_sample.shape[1] + (g_sample_index[index1] + 1)))
    total_omega = np.transpose(total_omega)

    return [total_omega, g_sample_temp, g_sample_index_temp]


def sing_crystal_rot_diff_exp(labsource, grain, hkl_list, omegabounds):
    # **********************************************************************************************
    # Name:    sing_crystal_rot_diff_exp
    # Purpose: function that simulates a rotating virtual diffraction experiment on a single crystal
    # Input:   labsource (object) - contains energy and incoming x-ray direction
    #          grain (object) - contains grain information
    #          hkl_list (m x 3) - contains m hkl plane vectors to interrogate for diffraction event
    #          omegabounds (tuple) - contains omega rotation bounds in tuple [low, high, stepSize]
    #                                in degrees
    # Output:  Function outputs a tuple in the form [two_theta, eta, k_out_lab, total_omega]
    #          two_theta (n x 1 matrix) - matrix of two_theta angles for n diffraction events
    #          eta (n x 1 matrix) - matrix of eta angles for n events
    #          k_out_lab (n x 3 matrix) - matrix of outgoing scattering vectors for n events (in lab
    #                                     coord system)
    #          total_omega (n x 1 matrix) - omega values for n diffraction events
    #          g_sample (3 x n matrix) - reciprocal lattice vectors for n events
    #          g_sample_index (n x 1 matrix) - gives indices of g_sample used in n events
    # Notes:   Units: Angstroms (unit cell and incoming wavelength) and degrees (unit cell, omega
    #          bounds, omega, two_theta, eta)
    # **********************************************************************************************

    # reciprocal lattice vectors (in columns of matrix) in sample coord system,
    # grain.reciprocal_strain is calculated in sample_class.py and accounts for zero strain input
    g_sample = calc_g_sample(grain, hkl_list)

    # calc omega values for rotating diffraction experiment
    [total_omega, g_sample, g_sample_index] = calc_omega_rot_diff_exp(g_sample, labsource.k_in_lab)

    # complete bounds control for omega bounds
    low_bounds_index = np.where(total_omega < omegabounds[0])[0]
    high_bounds_index = np.where(total_omega > omegabounds[1])[0]
    total_omega[low_bounds_index] = total_omega[low_bounds_index] + 360
    total_omega[high_bounds_index] = total_omega[high_bounds_index] - 360

    out_of_bounds_index = np.concatenate((np.where(total_omega < omegabounds[0])[0],
                                          np.where(total_omega > omegabounds[1])[0]))

    total_omega = np.delete(total_omega, out_of_bounds_index)
    g_sample = np.delete(g_sample, out_of_bounds_index, 1)
    g_sample_index = np.delete(g_sample_index, out_of_bounds_index)

    # build reciprocal lattice vector list in the lab coord system
    sin_omega = np.sin(np.deg2rad(total_omega))
    cos_omega = np.cos(np.deg2rad(total_omega))

    # create components of g_lab, reminder g_sample (0 = x, 1 = y, 2 = z)
    g_lab_x = np.multiply(cos_omega, g_sample[0, :]) + np.multiply(sin_omega, g_sample[2, :])
    g_lab_y = g_sample[1, :]
    g_lab_z = np.multiply(-sin_omega, g_sample[0, :]) + np.multiply(cos_omega, g_sample[2, :])
    g_lab = np.vstack((g_lab_x, g_lab_y, g_lab_z))

    # build outgoing wave vector list in the lab coord system, k_in_mat is a matrix of repeating k_
    # in vectors the size of g_lab for easy matrix addition
    k_in_mat = np.transpose(np.tile(labsource.k_in_lab, (np.shape(g_lab)[1], 1)))
    k_out_lab = k_in_mat + g_lab

    # get rid of bad solutions
    precis = 1E-10
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    mag_k_out = np.linalg.norm(k_out_lab, axis=0)

    # preparing outputs
    index = np.where(np.abs((mag_k_out - mag_k_in) / mag_k_in) < precis)
    total_omega = total_omega[index]
    g_sample = g_sample[:, index].reshape(3, -1)
    g_lab = g_lab[:, index].reshape(3, -1)
    k_out_lab = k_out_lab[:, index]
    k_out_lab = k_out_lab[:, 0]
    g_sample_index = g_sample_index[index]

    # determine two theta and eta
    mag_g = np.linalg.norm(g_sample, axis=0)
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    two_theta = np.rad2deg(2 * np.arcsin(np.divide(mag_g, (2 * mag_k_in))))
    eta = np.rad2deg(np.arctan2(k_out_lab[1, :], k_out_lab[0, :]))

    return [two_theta, eta, k_out_lab, total_omega, g_sample, g_sample_index]


def multi_crystal_rot_diff_exp(labsource, crystal_list, hkl_list, omegabounds):
    # **********************************************************************************************
    # Name:    multi_crystal_rot_diff_exp
    # Purpose: function that simulates a rotating virtual diffraction experiment on a multiple
    #          crystals
    # Input:   labsource (object) - contains energy and incoming x-ray direction
    #          crystal_list (m x 1) - list that contains m crystal objects for experiment
    #          hkl_list (n x 3) - contains n hkl plane vectors to interrogate for diffraction event
    #          omegabounds (tuple) - contains omega rotation bounds in tuple [low, high, stepSize]
    #                                in degrees
    # Output:  multi_crystal_list (m x 1) - Function outputs list of m tuples in the form
    #                                       [two_theta, eta, k_out_lab, total_omega]. One tuple for
    #                                       each crystal in crystal_list.
    # Notes:   Units: Angstroms (unit cell and incoming wavelength) and degrees (unit cell and omega
    #          bounds)
    # **********************************************************************************************

    multi_crystal_list = []

    for crystal in crystal_list:
        temp_tuple = sing_crystal_rot_diff_exp(labsource, crystal, hkl_list,
                                                                    omegabounds)
        multi_crystal_list.append(temp_tuple)

    return multi_crystal_list


def calc_crystal_pos_lab(p_sample, omega):
    # **********************************************************************************************
    # Name:    calc_crystal_position_lab
    # Purpose: function calculates crystal position vectors in lab during a rotating diffraction exp
    # Input:   p_sample (3 x 1 vector) - crystal position vector in the sample coord system
    #          omega (n x 1 vector) - omega values for n diffraction events
    # Output:  p_lab (3 x n matrix) - crystal position vectors in lab coord system for n diffraction
    #                                 events
    # Notes:   none
    # **********************************************************************************************

    # generate a n x 1 vector containing only the value one
    one_vector = np.ones(np.shape(omega)[0])

    # calculate n x 1 vectors that hold p_sample values
    p_sam_x = p_sample[0] * one_vector
    p_sam_y = p_sample[1] * one_vector
    p_sam_z = p_sample[2] * one_vector

    # calculate sin and cos for omegas
    sin = np.sin(np.deg2rad(omega))
    cos = np.cos(np.deg2rad(omega))

    # calculate components for p_lab
    p_lab_x = p_sam_x * cos + p_sam_z * sin
    p_lab_y = p_sam_y
    p_lab_z = -1 * p_sam_x * sin + p_sam_z * cos

    # combine components for p_lab
    p_lab = np.column_stack((p_lab_x, p_lab_y, p_lab_z))
    p_lab = np.transpose(p_lab)

    return p_lab


def sing_crystal_find_det_intercept_mesh(detector, mesh, k_out_lab, omega):
    # **********************************************************************************************
    # Name:    sing_crystal_find_det_intercept_mesh
    # Purpose: function that calculates detector intercepts from a single crystal virtual
    #          diffraction experiment using a mesh
    # Input:   detector (object) - holds all detector info
    #          mesh (object) - holds mesh info for grain
    #          k_out_lab (3 x n matrix) - holds outgoing wave vectors for n events in lab
    #                                     coord system
    #          omega (1 x n matrix) - holds omega values for n diffraction events
    # Output:  Function outputs a tuple in the form [zeta, zeta_pix, k_out_lab, omega]
    #          zeta (3 x p matrix) - holds the position vectors (x,y,z) of intercepts where p =
    #                                events occurred * mesh points
    #          zeta_pix (2 x p matrix) - holds the position vectors (x,y) of pixels on the detector
    #                                    where p = events occurred * mesh points
    #          n_k_out_lab (3 x p matrix) - holds outgoing wave vectors in lab coord system for
    #                                       each pixel intercepts p = events occurred * mesh points
    #          n_omega (1 x p matrix) - holds omega values for each pixel intercepts p = events
    #                                   occurred * mesh points
    # Notes:   none
    # **********************************************************************************************

    # initialize variables
    zeta = np.empty((3, 0))
    n_k_out_lab = np.empty((3, 0))
    n_omega = np.empty((0,))

    # Calculate magnitude and unit vectors for outgoing wavelength
    mag_k_out_lab = np.linalg.norm(k_out_lab, axis=0)
    mat_mag_k_out = np.tile(mag_k_out_lab, (3, 1))
    unit_k_out_lab = np.divide(k_out_lab, mat_mag_k_out)

    # iterate over each block in the mesh
    for i in range(np.shape(mesh.mesh)[0]):
        # calculate p_sample for this block and create n instances
        p_sample = mesh.grain.grainCOM + mesh.mesh[i, 0:3]

        # Get p_lab with call to calc_crystal_position_lab (3 x n matrix)
        p_lab = calc_crystal_pos_lab(p_sample, omega)

        # Calculate scale factor z
        z = -1 * np.divide(np.add(detector.distance, p_lab[2, :]), unit_k_out_lab[2, :])

        # Calculate intercept vectors
        temp_zeta = p_lab + np.multiply(np.tile(z, (3, 1)), unit_k_out_lab)
        zeta = np.hstack((zeta, temp_zeta))

        # Add values to new lists
        n_k_out_lab = np.hstack((n_k_out_lab, k_out_lab))
        n_omega = np.hstack((n_omega, omega))

    # Calculate intercept pixels
    zeta_pix = zeta * detector.pixel_density_mm
    zeta_pix = zeta_pix[0:2, :]

    return [zeta, zeta_pix, n_k_out_lab, n_omega]


def multi_crystal_find_det_intercept_mesh(detector, mesh_list, k_out_lab_list, omega_list):
    # **********************************************************************************************
    # Name:    multi_crystal_find_det_intercept_mesh
    # Purpose: function that calculates detector intercepts from a multi crystal virtual
    #          diffraction experiment
    # Input:   detector (object) - holds detector info
    #          mesh_list (m x 1 list) - holds m mesh objects
    #          k_out_lab_list (m x n x 3 matrix) -  holds m crystals outgoing wave vectors in the
    #                                               lab coord system for n diffraction events
    #          omega_list (m x n x 1 matrix) -  holds m crystals omega values for n diffraction
    #                                           events
    # Output:  Function outputs a tuple in the form [zeta, zeta_pix]
    #          zeta (3 x p matrix) - holds the position vectors (x,y,z) of intercepts where p
    #                                events occurred, p = m * n * mesh points
    #          zeta_pix (2 x p matrix) - holds the position vectors (x,y) of pixels on the detector
    #                                    where p events occurred, p = m * n * mesh points
    #          new_k_out_list (3 x p matrix) - holds outgoing vectors in lab coord system for p
    #                                          events, p = m * n * mesh points
    #          new_omega_list (1 x p matrix) - holds omega values for p events, p = m * n * mesh
    #                                          points
    # Notes:   none
    # **********************************************************************************************

    # initialize zeta and zeta_pix
    zeta = np.empty((3, 0))
    zeta_pix = np.empty((2, 0))
    new_k_out_list = np.empty((3, 0))
    new_omega_list = np.empty((0, ))

    # iterate over each mesh and call single_crystal_find_detector_intercept_mesh, add to lists
    for i in range(len(mesh_list)):
        k_out_lab = k_out_lab_list[i]
        mesh = mesh_list[i]
        omega = omega_list[i]
        [temp_zeta, temp_zeta_pix, n_k_out, n_omega] = \
            sing_crystal_find_det_intercept_mesh(detector, mesh, k_out_lab, omega)
        # add to lists
        zeta = np.hstack((zeta, temp_zeta))
        zeta_pix = np.hstack((zeta_pix, temp_zeta_pix))
        new_k_out_list = np.hstack((new_k_out_list, n_k_out))
        new_omega_list = np.hstack((new_omega_list, n_omega))

    return [zeta, zeta_pix, new_k_out_list, new_omega_list]


def sing_crystal_find_det_intercept(detector, p_sample, k_out_lab, omega):
    # **********************************************************************************************
    # Name:    sing_crystal_find_det_intercept
    # Purpose: function that calculates detector intercepts from a single crystal virtual
    #          diffraction experiment
    # Input:   detector (object) - holds all detector info
    #          k_out_lab (n x 3 matrix) - holds outgoing wave vectors for n events in lab
    #                                     coord system
    #          p_sample (3 x 1 vector) - holds crystal position vector for n events in sample coord
    #                                    system
    #          omega (n x 1 matrix) - holds omega values for n diffraction events
    # Output:  Function outputs a tuple in the form [zeta, zeta_pix]
    #          zeta (3 x n matrix) - holds the position vectors (x,y,z) of intercepts where n
    #                                events occurred
    #          zeta_pix (2 x n matrix) - holds the position vectors (x,y) of pixels on the detector
    #                                    where n events occurred
    # Notes:   none
    # **********************************************************************************************

    # Get p_lab with call to calc_crystal_position_lab (3 x n matrix)
    p_lab = calc_crystal_pos_lab(p_sample, omega)

    # Calculate magnitude and unit vectors for outgoing wavelength
    mag_k_out_lab = np.linalg.norm(k_out_lab, axis=0)
    mat_mag_k_out = np.tile(mag_k_out_lab, (3, 1))
    unit_k_out_lab = np.divide(k_out_lab, mat_mag_k_out)

    # Calculate scale factor z
    z = -1 * np.divide(np.add(detector.distance, p_lab[2, :]), unit_k_out_lab[2, :])

    # Calculate intercept vectors
    zeta = p_lab + np.multiply(np.tile(z, (3, 1)), unit_k_out_lab)

    # Calculate intercept pixels
    zeta_pix = zeta * detector.pixel_density_mm
    zeta_pix = zeta_pix[0:2, :]

    return [zeta, zeta_pix]


def multi_crystal_find_det_intercept(detector, crystal_list, k_out_lab_list, omega_list):
    # **********************************************************************************************
    # Name:    multi_crystal_find_det_intercept
    # Purpose: function that calculates detector intercepts from a multi crystal virtual
    #          diffraction experiment
    # Input:   detector (object) - holds detector info
    #          crystal_list (m x 1 list) - holds m crystal objects
    #          k_out_lab_list (m x n x 3 matrix) -  holds m crystals outgoing wave vectors in the
    #                                               lab coord system for n diffraction events
    #          omega_list (m x n x 1 matrix) -  holds m crystals omega values for n diffraction
    #                                           events
    # Output:  Function outputs a tuple in the form [zeta, zeta_pix]
    #          zeta (3 x p matrix) - holds the position vectors (x,y,z) of intercepts where p
    #                                events occurred, p = m * n
    #          zeta_pix (2 x p matrix) - holds the position vectors (x,y) of pixels on the detector
    #                                    where p events occurred, p = m * n
    #          flat_omega_list (p x 1 matrix) = holds omega values for p events, p = m * n
    # Notes:   none
    # **********************************************************************************************

    # initialize zeta and zeta_pix
    zeta = np.empty([3, 0])
    zeta_pix = np.empty([2, 0])
    new_omega_list = np.empty([0, ])

    # iterate over each crystal and call single_crystal_find_detector_intercept, add to total lists
    for i in range(len(crystal_list)):
        k_out_lab = k_out_lab_list[i]
        p_sample = crystal_list[i].grainCOM
        omega = omega_list[i]
        [temp_zeta, temp_zeta_pix] = sing_crystal_find_det_intercept(detector, p_sample,
                                                                     k_out_lab, omega)
        # add to lists
        zeta = np.hstack((zeta, temp_zeta))
        zeta_pix = np.hstack((zeta_pix, temp_zeta_pix))
        new_omega_list = np.hstack((new_omega_list, omega))

    return [zeta, zeta_pix, new_omega_list]


def display_detector(detector, zeta_pix, circle=False, radius=1000):
    # **********************************************************************************************
    # Name:    detector_intercept
    # Purpose: function that displays virtual diffraction image
    # Input:   detector (object) - holds all detector info
    #          zeta_pix (n x 2 matrix) - holds the position vectors (x,y) of pixels on the detector
    #                                    where n events occurred
    # Output:  Scatter plot image of the diffraction events
    # Notes:   none
    # **********************************************************************************************

    # grabbing all x and y values, adding a point for the transmitted incoming at the origin
    all_x = np.append(zeta_pix[0, :].tolist(), [0])
    all_y = np.append(zeta_pix[1, :].tolist(), [0])

    # plot using a scatter
    fig, ax = plot.subplots(nrows=1, ncols=1)
    ax.scatter(all_x, all_y, marker=".", color="white", s=3)
    # adding circle to plot
    if circle:
        [circle_x, circle_y] = add_circle([], [], radius=radius)
        ax.scatter(circle_x, circle_y, marker=".", color="red", s=1)
    ax.set_facecolor('xkcd:black')
    plot.xlim(-detector.width / 2, detector.width / 2)
    plot.ylim(-detector.height / 2, detector.height / 2)
    plot.show()

    return 0


def display_detector_bounded(detector, zeta_pix, omega, omega_bounds, circle=False, radius=1000):
    # **********************************************************************************************
    # Name:    display_detector_bounded
    # Purpose: function that displays virtual diffraction image, bounded by omegas
    # Input:   detector (object) - holds all detector info
    #          zeta_pix (n x 2 matrix) - holds the position vectors (x,y) of pixels on the detector
    #                                    where n events occurred
    #          omega (n x 1) matrix - holds the omega values for n diffraction events
    #          omega_bounds (tuple) - holds bounds for omega and thresholds what events to display,
    #                                 takes the form [omega_low, omega_high]
    # Output:  Scatter plot image of the diffraction events
    # Notes:   none
    # **********************************************************************************************

    # initializing x and y to empty matrix
    x = np.empty((0, 1))
    y = np.empty((0, 1))

    # grabbing all x and y values
    all_x = zeta_pix[0, :].tolist()
    all_y = zeta_pix[1, :].tolist()

    # add events for omega values inside the bounds
    for i in range(np.shape(omega)[0]):
        if omega_bounds[0] < omega[i] < omega_bounds[1]:
            x = np.append(x, all_x[0][i])
            y = np.append(y, all_y[0][i])

    # adding a point for the transmitted incoming wavelength at the origin
    x = np.append(x, [0])
    y = np.append(y, [0])

    # plot using a scatter
    fig, ax = plot.subplots()
    ax.scatter(x, y, marker=".", color="white", s=3)
    ax.set_facecolor('xkcd:black')
    ax.set_xlim(-detector.width / 2, detector.width / 2)
    ax.set_ylim(-detector.height / 2, detector.height / 2)
    plot.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the left edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False)  # labels along the left edge are off

    # adding circle to plot
    if circle:
        [circle_x, circle_y] = add_circle([], [], radius=radius)
        ax.scatter(circle_x, circle_y, marker=".", color="red", s=1)

    # show plot
    plot.show()

    return 0


def display_detector_bounded_animate(detector, zeta_pix, omega, omega_bounds):
    # **********************************************************************************************
    # Name:    display_detector_bounded_animate
    # Purpose: function that animates virtual diffraction images, as sample is rotated by variables
    #          in omega_bounds
    # Input:   detector (object) - holds all detector info
    #          zeta_pix (n x 2 matrix) - holds the position vectors (x,y) of pixels on the detector
    #                                    where n events occurred
    #          omega (n x 1) matrix - holds the omega values for n diffraction events
    #          omega_bounds (tuple) - holds bounds for omega and thresholds what events to display,
    #                                 takes the form [omega_low, omega_high, omega_step_size]
    # Output:  Animated scatter plot image of the diffraction events
    # Notes:   none
    # **********************************************************************************************

    # initializing omega variables
    omega_low = omega_bounds[0]
    omega_high = omega_bounds[1]
    omega_step_size = omega_bounds[2]
    omega_steps = (omega_high - omega_low) / omega_step_size

    # grabbing all x and y values
    all_x = zeta_pix[0, :].tolist()
    all_y = zeta_pix[1, :].tolist()

    # initialize animation plots
    fig, ax = plot.subplots()
    scat = ax.scatter([], [], marker=".", color="white", s=3)

    # define init function for plot
    def init():
        ax.set_facecolor('xkcd:black')
        ax.set_xlim(-detector.width / 2, detector.width / 2)
        ax.set_ylim(-detector.height / 2, detector.height / 2)
        plot.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the left edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False)  # labels along the left edge are off
        return scat,

    # define update function for plot
    def update(frame):
        # initializing x and y to empty matrix
        x = np.empty((0, 1))
        y = np.empty((0, 1))

        # add events for omega values inside the bounds
        low = frame - omega_step_size / 2
        high = frame + omega_step_size / 2
        for i in range(np.shape(omega)[0]):
            if low < omega[i] < high:
                x = np.append(x, all_x[0][i])
                y = np.append(y, all_y[0][i])

        # adding a point for the transmitted incoming wavelength at the origin
        x = np.append(x, [0])
        y = np.append(y, [0])
        x = x.reshape((np.shape(x)[0], 1))
        y = y.reshape((np.shape(y)[0], 1))

        # set data for plot
        data = np.hstack((x, y))
        scat.set_offsets(data)
        return scat,

    ani = plot_ani.FuncAnimation(fig, update,
                                 frames=np.linspace(omega_low, omega_high, omega_steps),
                                 init_func=init, blit=True, repeat=False)
    plot.show()

    return 0


def add_circle(x, y, radius=1, points=100):
    # **********************************************************************************************
    # Name:    add_circle
    # Purpose: function that adds a circle to the virtual diffraction image
    # Input:   x (n x 1) - holds x positions for n diffraction events on the detector
    #          y (n x 1) - holds y positions for n diffraction events on the detector
    #          radius (int) - the radius of the circle on the detector
    #          points (int) - how many points to display of the circle
    # Output:  [x, y] - where the circle coordinates have been appended to the x, y lists
    # Notes:   none
    # **********************************************************************************************

    theta_step = 2 * sciconst.pi / points

    for i in range(points):
        x = np.append(x, [radius * np.cos(theta_step * i)])
        y = np.append(y, [radius * np.sin(theta_step * i)])

    return [x, y]
