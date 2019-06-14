# **************************************************************************************************
# Name:    sing_crystal_rot_virt_diff_test.py
# Purpose: Test the single_crystal functions in virtual_diffractometer_functions.py
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

# Grain_1 parameters (unitCell, dimension, COM, orientation, intensity)
quat_1 = np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3])
Grain_1 = Grain(unitCell_1, np.array([1.0, 1.0, 1.0]), np.array([0, 0, 0]), quat_1, 100)

# Sample_1 parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
omegaLow = 0
omegaHigh = 180
Sample_1 = Sample(np.array([Grain_1]), omegaLow, omegaHigh, 1)


# ************************************* Test Function Definition ***********************************
def test():
    # initialize hkl vectors and omega_bounds
    path = "C:\Git Repositories\VirtualDiffractometer\data\hkl_list_3.csv"
    hkl_list = read_hkl_from_csv(path)
    hkl_list = gen_hkl_fam_from_list(hkl_list, cubic=True)

    omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]

    print("Starting #1")
    # call single crystal rotating diffraction experiment, time is for measuring calculation length
    t = time.time()
    [two_theta, eta, k_out_lab, omega] = sing_crystal_rot_diff_exp(LabSource_1, Sample_1.grains[0],
                                                                   hkl_list, omega_bounds)
    print("#1 Elapsed: ", time.time() - t)

    # call single crystal intercept
    [zeta, zeta_pix] = sing_crystal_find_det_intercept(Detector_1, Sample_1.grains[0].grainCOM,
                                                       k_out_lab, omega)

    # call display detector to display a diffraction image
    display_omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh]
    display_detector(Detector_1, zeta_pix, omega, display_omega_bounds)

    print("two_theta: ", np.shape(two_theta))
    print("eta: ", np.shape(eta))
    print("k_out_lab: ", np.shape(k_out_lab))
    print("omega: ", np.shape(omega))

    return 0


# ************************************* Test Function Execution ************************************
test()
