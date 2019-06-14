# **************************************************************************************************
# Name:    multi_crystal_rot_virt_diff_test.py
# Purpose: Test the multi_cyrstal functions in virtual_diffractometer_functions.py
# Input:   Listed below under "Variable Definitions" section
# Output:  Diffraction image
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
from classes.equipment_class import Detector, LabSource
from classes.sample_class import UnitCell, Grain, Sample
from utils.sample_functions import read_hkl_from_csv, gen_hkl_fam_from_list
from utils.virtual_diffractometer_functions import (sing_crystal_rot_diff_exp,
                                                    multi_crystal_rot_diff_exp,
                                                    sing_crystal_find_det_intercept,
                                                    multi_crystal_find_det_intercept,
                                                    display_detector)
import numpy as np
import time

# *************************************** Variable Definitions *************************************

# Detector_1 parameters (distance, width, height, pixel_density)(mm, pixel, pixel, pixel/mm)
detector_size = 10000
Detector_1 = Detector(1000, detector_size, detector_size, 10)

# LabSource_1 parameters (energy, incomingXray) (keV, unitVector)
LabSource_1 = LabSource(55.618, LabSource.lab_z)

# unitCell_1 lattice parameters (a, b, c, alpha, beta, gamma) (Angstroms, degrees)
unitCell_1 = UnitCell(np.array([2, 2, 2, 90, 90, 90]))
unitCell_2 = UnitCell(np.array([4, 4, 4, 90, 90, 90]))

quat_1 = np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3])
quat_2 = np.array([1.0, 0, 0, 0])
quat_3 = np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3])
# Grain_1 parameters (unitCell, dimension, COM, orientation, intensity)
Grain_1 = Grain(unitCell_2, np.array([1.0, 1.0, 1.0]), np.array([0, 0, 0]), quat_1, 100)
Grain_2 = Grain(unitCell_2, np.array([1.0, 1.0, 1.0]), np.array([5, 0, 0]), quat_2, 100)
Grain_3 = Grain(unitCell_2, np.array([1.0, 1.0, 1.0]), np.array([-5, 0, 0]), quat_3, 100)

# Sample_1 parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
omegaLow = 0
omegaHigh = 180
Sample_1 = Sample(np.array([Grain_1, Grain_2, Grain_3]), omegaLow, omegaHigh, 1)


# ************************************* Test Function Definition ***********************************
def test():
    # initialize hkl vectors and omega_bounds
    path = "C:\Git Repositories\VirtualDiffractometer\data\hkl_list_3.csv"
    hkl_list = read_hkl_from_csv(path)
    hkl_list = gen_hkl_fam_from_list(hkl_list, cubic=True)

    omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]

    print("Starting #1")
    # call multi_crystal_diff, time is for measuring length of calculation
    t = time.time()
    multi_rot_diff_list = multi_crystal_rot_diff_exp(LabSource_1, Sample_1.grains,
                                                     hkl_list, omega_bounds)
    print("#1 Elapsed: ", time.time() - t)

    # call multi_crystal_intercept, takes list of p_sample, k_out, omega for each crystal
    p_sample_list = []
    k_out_list = []
    omega_list = []

    for i in range(len(Sample_1.grains)):
        p_sample_list.append(Sample_1.grains[i])
        k_out_list.append(multi_rot_diff_list[i][2])
        omega_list.append(multi_rot_diff_list[i][3])
    [zeta, zeta_pix, new_omega] = multi_crystal_find_det_intercept(Detector_1, p_sample_list,
                                                                        k_out_list, omega_list)

    # call display_detector for generating a diffraction image
    display_omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh]
    display_detector(Detector_1, zeta_pix, new_omega, display_omega_bounds)

    return 0


# ************************************* Test Function Execution ************************************
test()
