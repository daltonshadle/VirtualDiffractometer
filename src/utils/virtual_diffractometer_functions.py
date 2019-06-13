# **************************************************************************************************
# Name:    virtual_diffractometer_functions.py
# Purpose: Declaration and implementation of virtual diffractometer functions
# Input:   none
# Output:  Function definitions for virtual diffractometer experiment
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
from classes.equipment_class import LabSource, Detector
from classes.sample_class import Grain, UnitCell, Sample
import numpy as np
import matplotlib.pyplot as plot
import sympy
import scipy.constants as sciconst


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

    # compute magnitude of reciprocal-lattice vectors, reshape to (m x 1 matrix) where m = # of hkl
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

    # compute omega values, stack corresponding values, omega is in radians
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
    # Notes:   Units: Angstroms (unit cell and incoming wavelength) and degrees (unit cell and omega
    #          bounds)
    # **********************************************************************************************

    # reciprocal lattice vectors (in columns of matrix) in sample coord system
    g_sample = (grain.quat2rotmat() * grain.unitCell.get_reciprocal_lattice_vectors()
               * np.transpose(hkl_list))

    # split into x, y, z in the sample coord system
    g_sample_x = np.transpose(g_sample[0, :])
    g_sample_y = np.transpose(g_sample[1, :])
    g_sample_z = np.transpose(g_sample[2, :])

    # set up quadratic formula for solving omega values: qf = (-b +- sqrt(b^2 - 4*a*c)) / (2*a)
    Y = ((np.power(g_sample_x, 2) + np.power(g_sample_y, 2) + np.power(g_sample_z, 2))
         / (2 * np.linalg.norm(labsource.k_in_lab)))
    a = np.power(g_sample_x, 2) + np.power(g_sample_z, 2)
    b = 2 * np.multiply(Y, g_sample_x)
    c = np.power(Y, 2) - np.power(g_sample_z, 2)
    rad = np.sqrt((np.power(b, 2) - 4 * np.multiply(a, c)))

    # find solution indices for quadratic formula (index1 = 1 solution & index2 = 2 solutions)
    precis = 1E-6
    index1 = np.where(np.abs(rad < precis))[0]
    index2 = np.where(np.abs(rad >= precis))[0]

    # calculate solution of quadratic formula and find omega values (in degrees)
    plus_solution = np.divide((-np.take(b, index2) + np.take(rad, index2)), (2 * np.take(a, index2)))
    minus_solution = np.divide((-np.take(b, index2) - np.take(rad, index2)), (2 * np.take(a, index2)))
    zero_solution = np.divide(-np.take(b, index1), (2 * np.take(a, index1)))

    plus_omega = np.array(np.rad2deg(np.real(np.arcsin(plus_solution)))).flatten()
    minus_omega = np.array(np.rad2deg(np.real(np.arcsin(minus_solution)))).flatten()
    zero_omega = np.array(np.rad2deg(np.real(np.arcsin(zero_solution)))).flatten()

    # gather all omega solutions and reciprocal lattice vectors in the sample frame
    total_omega = np.empty((0,), float)
    g_sample_temp = np.empty((3, 0), float)

    # plus omega solutions
    total_omega = np.append(total_omega, plus_omega, axis=0)
    total_omega = np.append(total_omega, np.add(-plus_omega, 180), axis=0)
    g_sample_temp = np.concatenate(
        (g_sample_temp, g_sample[:, index2], g_sample[:, index2]), axis=1)
    # minus omega solutions
    total_omega = np.append(total_omega, minus_omega, axis=0)
    total_omega = np.append(total_omega, np.add(-minus_omega, -180), axis=0)
    g_sample_temp = np.concatenate(
        (g_sample_temp, g_sample[:, index2], g_sample[:, index2]), axis=1)
    # zero omega solutions
    total_omega = np.append(total_omega, zero_omega, axis=0)
    g_sample_temp = np.concatenate((g_sample_temp, g_sample[:, index1]), axis=1)

    # complete bounds control for omega bounds
    out_of_bounds_index = np.concatenate((np.where(total_omega < omegabounds[0])[0],
                                          np.where(total_omega > omegabounds[1])[0]))

    total_omega = np.delete(total_omega, out_of_bounds_index)
    g_sample = np.delete(g_sample_temp, out_of_bounds_index, 1)

    # build reciprocal lattice vector list in the lab coord system
    sin_omega = np.sin(np.deg2rad(total_omega))
    cos_omega = np.cos(np.deg2rad(total_omega))
    g_lab = np.empty((3, 0), float)

    for y in range(0, len(total_omega)):
        rotmat_L2C = np.matrix([[cos_omega[y],  0, sin_omega[y]],
                                [0,             1,           0],
                                [-sin_omega[y], 0, cos_omega[y]]])

        temp_g_lab = rotmat_L2C * g_sample[:, y]
        g_lab = np.append(g_lab, temp_g_lab, axis=1)

    # build outgoing wave vector list in the lab coord system
    k_in_mat = np.ones(np.shape(g_lab))
    for i in range(0, 3):
        k_in_mat[i, :] = k_in_mat[i, :] * labsource.k_in_lab[i]

    k_out_lab = k_in_mat + g_lab
    #print(k_in_mat, "\n", g_lab)

    # get rid of bad solutions
    precis = 1E-10
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    mag_k_out = np.linalg.norm(k_out_lab, axis=0)

    index = np.where(np.abs((mag_k_out - mag_k_in) / mag_k_in) < precis)
    total_omega = total_omega[index]
    g_sample = g_sample[:, index]
    g_lab = g_lab[:, index]
    k_out_lab = k_out_lab[:, index]
    k_out_lab = k_out_lab[:, 0]

    # determine two theta and eta
    mag_g = np.linalg.norm(g_sample, axis=0)
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    two_theta = np.rad2deg(2 * np.arcsin(np.divide(mag_g, (2 * mag_k_in))))
    eta = np.rad2deg(np.arctan2(k_out_lab[1, :], k_out_lab[0, :]))

    return [two_theta, eta, k_out_lab, total_omega]


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


def compute_crystal_pos_lab(p_sample, omega):
    # **********************************************************************************************
    # Name:    compute_crystal_position_lab
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

    # Get p_lab with call to compute_crystal_position_lab (3 x n matrix)
    p_lab = compute_crystal_pos_lab(p_sample, omega)

    # Calculate magnitude and unit vectors for outgoing wavelength
    mag_k_out_lab = np.linalg.norm(k_out_lab, axis=0)
    mat_k_out = np.tile(mag_k_out_lab, (3, 1))
    unit_k_out_lab = np.divide(k_out_lab, mat_k_out)

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


def display_detector(detector, zeta_pix, omega, omega_bounds, circle=False, radius=1000):
    # **********************************************************************************************
    # Name:    detector_intercept
    # Purpose: function that displays virtual diffraction image
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
    fig, ax = plot.subplots(nrows=1, ncols=1)
    ax.scatter(x, y, marker=".", color="white", s=3)
    # adding circle to plot
    if circle:
        [circle_x, circle_y] = add_circle([], [], radius=radius)
        ax.scatter(circle_x, circle_y, marker=".", color="red", s=1)
    ax.set_facecolor('xkcd:black')
    plot.xlim(-detector.width / 2, detector.width / 2)
    plot.ylim(-detector.height / 2, detector.height / 2)
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
