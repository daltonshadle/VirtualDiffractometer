# **************************************************************************************************
# Name:    virt_diff_test.py
# Purpose: Test the function in virtual_diffractometer_functions.py
# Input:   Listed below under "Variable Definitions" section
# Output:  Diffraction image
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
from classes.equipment_class import Detector, LabSource
from classes.sample_class import UnitCell, Grain, Sample
from utils.sample_functions import read_hkl_from_csv, gen_hkl_fam_from_list
from utils.virtual_diffractometer_functions import (multi_crystal_find_detector_intercept,
                                                    display_detector_intercept,
                                                    multi_crystal_rotating_diffraction_experiment)
import numpy as np

# *************************************** Variable Definitions *************************************

# Detector_1 parameters (distance, width, height, pixel_density)(mm, pixel, pixel, pixel/mm)
detector_size = 10000
Detector_1 = Detector(1000, detector_size, detector_size, 10)

# LabSource_1 parameters (energy, incomingXray) (keV, unitVector)
LabSource_1 = LabSource(55.618, LabSource.lab_z)

# unitCell_1 lattice parameters (a, b, c, alpha, beta, gamma) (Angstroms, degrees)
unitCell_1 = UnitCell(np.array([3, 3, 3, 90, 90, 90]))
unitCell_2 = UnitCell(np.array([1, 1, 1, 90, 90, 90]))

# Grain_1 parameters (unitCell, dimension, COM, orientation, intensity)
Grain_1 = Grain(unitCell_1, np.array([1.0, 1.0, 1.0]), np.array([2, 0.5, 0.0]),
                np.array([1, .1, .1, .1]), 100)
Grain_1.rotmat2quat(np.eye(3))

Grain_2 = Grain(unitCell_2, np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.5, 0.0]),
                np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3]), 100)

chi = np.deg2rad(45); sinC = np.sin(chi); cosC = np.cos(chi)
theta = np.deg2rad(45); sinT = np.sin(theta); cosT = np.cos(theta)
my_mat = np.matrix([[cosT,           0,            sinT],
                    [sinC * sinT,   cosC,  -sinC * cosT],
                    [-cosC * sinT,  sinC,  cosC * cosT]])
Grain_2.rotmat2quat(my_mat)

# Sample_1 parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
Sample_1 = Sample(np.array([Grain_1]), 0, 180, 1)


# ************************************* Test Function Definition ***********************************
def test():
    path = "C:\Git Repositories\VirtualDiffractometer\data\hkl_list_10.csv"
    hkl_list = read_hkl_from_csv(path)

    omega_tuple = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]

    # call multi_crystal, returns list of tuples [two_theta, eta, k_out_lab, total_omega]
    multi_crystal_list = multi_crystal_rotating_diffraction_experiment(LabSource_1, Sample_1.grains,
                                                                       hkl_list, omega_tuple)

    # call multi_crystal, takes list of p_sample, k_out, omega for each crystal
    p_sample_list = []
    k_out_list = []
    omega_list = []

    for i in range(len(Sample_1.grains)):
        p_sample_list.append(Sample_1.grains[i])
        k_out_list.append(multi_crystal_list[i][2])
        omega_list.append(multi_crystal_list[i][3])
    [zeta, zeta_pix, new_omega] = multi_crystal_find_detector_intercept(Detector_1, p_sample_list,
                                                                        k_out_list, omega_list)

    display_omega_bounds = [78, 80]
    display_detector_intercept(Detector_1, zeta_pix, new_omega, display_omega_bounds)

    return 0


# ************************************* Test Function Execution ************************************
test()
