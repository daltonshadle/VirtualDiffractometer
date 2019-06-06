# **************************************************************************************************
# Name:    VirtualDiffractometer.py.py
# Purpose: Produces synthetic diffraction patterns and tiff files
# Input:   Listed below under "Variable Definitions" section
# Output:  Diffraction image tiff file
# Notes:   - Ideal detector parameters only (i.e., only energy and
#            sample-to-detector distance--no tilt, etc. )
#          - Doesn't consider intensity parameters (e.g., structure factor)
# **************************************************************************************************

# ********************************************* Imports ********************************************
from VirtDiffTools import *


# *************************************** Variable Definitions *************************************

# Detector_1 parameters (distance, energy, width, height)(mm, keV, pixel*e-4, pixel*e-4)
detector_size = 100
Detector_1 = Detector(1012.36, detector_size, detector_size)

# LabSource_1 parameters (energy, incomingXray) (keV, unitVector)
LabSource_1 = LabSource(55.618, LabSource.lab_z)

# unitCell_1 lattice parameters (a,b,c,alpha,beta,gamma) (Angstroms, degrees)
unitCell_1 = UnitCell(np.array([2, 2, 2, 90, 90, 90]))

# Grain_1 parameters (unitCell, dimension, COM, orientation, intensity)
Grain_1 = Grain(unitCell_1, np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.5, 0.0]),
                np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3]), 100)

# Sample_1 parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
Sample_1 = Sample(np.array([Grain_1]), 0, 180, 1)

# hkl_list initialization
hkl_list = np.matrix([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 1],
                      [1, 1, 2],
                      [1, 1, 3],
                      [1, 1, 4]])


# *************************************** Function Definitions *************************************
def virtual_diffractometer(labsource, grain, hkl_list, omegabounds):
    # **********************************************************************************************
    # Name:    virtual_diffractometer
    # Purpose: function that simulates a rotating virtual diffraction experiment on a single sample
    # Input:   labsource (object) - contains energy and incoming x-ray direction
    #          grain (object) - contains grain information
    #          hkl_list (list) - contains all hkl plains to interrogate for a diffraction event
    #          omegabounds (tuple) - contains omega rotation bounds in tuple [low, high, stepSize]
    #                                in degrees
    # Output:  two_theta (n x 1 matrix) - matrix of two theta angles for n diffraction events
    #          eta (n x 1 matrix) - matrix of eta angles for n events
    #          k_out_lab (n x 3 matrix) - matrix of outgoing scattering vectors for n events
    #          omega (n x 1 matrix) - omega values for n events
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

    # set up quadratic formula for solving omega values
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
    plusSolution = np.divide((-np.take(b, index2) + np.take(rad, index2)), (2 * np.take(a, index2)))
    minusSolution = np.divide((-np.take(b, index2) - np.take(rad, index2)), (2 * np.take(a, index2)))
    zeroSolution = np.divide(-np.take(b, index1), (2 * np.take(a, index1)))

    plusOmega = np.array(np.rad2deg(np.real(np.arcsin(plusSolution)))).flatten()
    minusOmega = np.array(np.rad2deg(np.real(np.arcsin(minusSolution)))).flatten()
    zeroOmega = np.array(np.rad2deg(np.real(np.arcsin(zeroSolution)))).flatten()

    # gather all omega solutions and reciprocal lattice vectors in the sample frame
    totalOmega = np.empty((0,), float)
    g_sample_temp = np.empty((3, 0), float)

    # plus omega solutions
    totalOmega = np.append(totalOmega, plusOmega, axis=0)
    totalOmega = np.append(totalOmega, np.add(-plusOmega, 180), axis=0)
    g_sample_temp = np.concatenate(
        (g_sample_temp, g_sample[:, index2], g_sample[:, index2]), axis=1)
    # minus omega solutions
    totalOmega = np.append(totalOmega, minusOmega, axis=0)
    totalOmega = np.append(totalOmega, np.add(-minusOmega, -180), axis=0)
    g_sample_temp = np.concatenate(
        (g_sample_temp, g_sample[:, index2], g_sample[:, index2]), axis=1)
    # zero omega solutions
    totalOmega = np.append(totalOmega, zeroOmega, axis=0)
    g_sample_temp = np.concatenate((g_sample_temp, g_sample[:, index1]), axis=1)

    # complete bounds control for omega bounds
    out_of_bounds_index = np.concatenate((np.where(totalOmega < omegabounds[0])[0],
                                          np.where(totalOmega > omegabounds[1])[0]))

    totalOmega = np.delete(totalOmega, out_of_bounds_index)
    g_sample = np.delete(g_sample_temp, out_of_bounds_index, 1)

    # build reciprocal lattice vector list in the lab coord system
    sinOmega = np.sin(np.deg2rad(totalOmega))
    cosOmega = np.cos(np.deg2rad(totalOmega))
    g_lab = np.empty((3, 0), float)

    for y in range(0, len(totalOmega)):
        rotmat_L2C = np.matrix([[cosOmega[y],  0, sinOmega[y]],
                                [0,            1,           0],
                                [-sinOmega[y], 0, cosOmega[y]]])

        temp_g_lab = rotmat_L2C * g_sample[:, y]
        g_lab = np.append(g_lab, temp_g_lab, axis=1)

    # build outgoing wave vector list in the lab coord system
    k_in_mat = np.ones(np.shape(g_lab))
    for i in range(0, 3):
        k_in_mat[i, :] = k_in_mat[i, :] * labsource.k_in_lab[i]

    k_out_lab = k_in_mat + g_lab

    # get rid of bad solutions
    precis = 1E-10
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    mag_k_out = np.linalg.norm(k_out_lab, axis=0)

    index = np.where(np.abs((mag_k_out - mag_k_in) / mag_k_in) < precis)
    totalOmega = totalOmega[index]
    g_sample = g_sample[:, index]
    g_lab = g_lab[:, index]
    k_out_lab = k_out_lab[:, index]
    k_out_lab = k_out_lab[:, 0]

    # determine two theta and eta
    mag_g = np.linalg.norm(g_sample, axis=0)
    mag_k_in = np.linalg.norm(labsource.k_in_lab)
    two_theta = np.rad2deg(2 * np.arcsin(np.divide(mag_g, (2 * mag_k_in))))
    eta = np.rad2deg(np.arctan2(k_out_lab[1, :], k_out_lab[0, :]))

    return [two_theta, eta, k_out_lab, totalOmega]


def detector_intercept(detector, k_out_lab):
    # **********************************************************************************************
    # Name:    detector_intercept
    # Purpose: function that simulates a rotating virtual diffraction experiment on a single sample
    # Input:   detector (object) - holds all detector info
    #          k_out_lab (n x 3 matrix) - holds all the outgoing wave vectors for n events
    # Output:  zeta (n x 2 matrix) - holds the intensity values of the detector as an
    #                                (x,y) array of points where n events occurred
    # Notes:   none
    # **********************************************************************************************

    x = -1 * np.divide(k_out_lab[0, :], k_out_lab[2, :])
    y = -1 * np.divide(k_out_lab[1, :], k_out_lab[2, :])

    zeta = np.vstack((x, y)) * detector.distance

    return zeta


def display_detector_intercept(detector, zeta):
    # **********************************************************************************************
    # Name:    detector_intercept
    # Purpose: function that simulates a rotating virtual diffraction experiment on a single sample
    # Input:   detector (object) - holds all detector info
    #          zeta (n x 2 matrix) - holds the intensity values of the detector as an
    #                                (x,y) array of points where n events occurred
    # Output:  none
    # Notes:   none
    # **********************************************************************************************

    x = zeta[0, :].tolist()
    y = zeta[1, :].tolist()
    print(x)
    print(y)
    plot.scatter(x, y, marker=5)
    i = 100
    plot.xlim(-i, i)
    plot.ylim(-i, i)
    plot.show()

    return 0


# ************************************* Main Function Definition ***********************************
def main():
    omega_tuple = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]
    [two_theta, eta, k_out_lab, omega] = virtual_diffractometer(LabSource_1, Sample_1.grains[0],
                                                                hkl_list, omega_tuple)
    zeta = detector_intercept(Detector_1, k_out_lab)
    display_detector_intercept(Detector_1, zeta)
    return 0


# ************************************* Main Function Execution ************************************
main()