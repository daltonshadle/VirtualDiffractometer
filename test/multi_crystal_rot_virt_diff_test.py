# **************************************************************************************************
# Name:    multi_crystal_rot_virt_diff_test.py
# Purpose: Test the multi_cyrstal functions in virtual_diffractometer_functions.py
# Input:   Listed below under "Variable Definitions" section
# Output:  Diffraction image
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
from classes.equipment_class import Detector, LabSource
from classes.sample_class import UnitCell, Grain, Sample, Mesh
from utils.sample_functions import (read_hkl_from_csv, gen_hkl_fam_from_list,
                                    read_grains_from_csv, mesh_list_from_grain_list)
from utils.virtual_diffractometer_functions import (multi_crystal_rot_diff_exp,
                                                    multi_crystal_find_det_intercept,
                                                    multi_crystal_find_det_intercept_mesh,
                                                    display_detector,
                                                    display_detector_bounded,
                                                    display_detector_bounded_animate)
from utils.math_functions import normalize
import numpy as np
import time
import os

# *************************************** Variable Definitions *************************************

# Detector parameters (distance, width, height, pixel_density)(mm, pixel, pixel, pixel/mm)
detector_size = 20000
Detector_1 = Detector(1000, detector_size, detector_size, 10)

# LabSource parameters (energy, incomingXray) (keV, unitVector)
LabSource_1 = LabSource(55.618, LabSource.lab_z)


# ************************************* Test Function Definition ***********************************
def test():
    # initialize grain and mesh list
    mesh_size = 1
    path = os.getcwd()
    path = path = path.split("src")[0] + "data\\multi_grains_15.csv"  # windows
    grain_list = read_grains_from_csv(path)
    mesh_list = mesh_list_from_grain_list(grain_list, mesh_size)

    # Sample parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
    omegaLow = 0
    omegaHigh = 180
    omegaStepSize = 1
    Sample_1 = Sample(np.array(grain_list), omegaLow, omegaHigh, omegaStepSize, np.array(mesh_list))

    # initialize hkl vectors and omega_bounds
    path = os.getcwd()
    path = path.split("src")[0] + "data\\hkl_list_10.csv"  # windows
    hkl_list = read_hkl_from_csv(path)

    omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]
    display_omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]

    print("Starting #1")
    # call multi_crystal_diff, time is for measuring length of calculation
    t = time.time()
    multi_rot_diff_list = multi_crystal_rot_diff_exp(LabSource_1, Sample_1.grains,
                                                     hkl_list, omega_bounds)
    print("#1 Elapsed: ", time.time() - t)

    # call multi_crystal_intercept, takes list of p_sample, k_out, omega for each crystal
    p_sample_list = []
    mesh_list = []
    k_out_list = []
    omega_list = []

    for i in range(len(Sample_1.grains)):
        p_sample_list.append(Sample_1.grains[i])
        mesh_list.append(Sample_1.meshes[i])
        k_out_list.append(multi_rot_diff_list[i][2])
        omega_list.append(multi_rot_diff_list[i][3])

    [zeta, zeta_pix, new_k_out, new_omega] = \
        multi_crystal_find_det_intercept_mesh(Detector_1, mesh_list, k_out_list, omega_list)

    # call display_detector for generating a diffraction image
    display_detector_bounded(Detector_1, zeta_pix, new_omega, display_omega_bounds)

    [zeta, zeta_pix, new_omega] = multi_crystal_find_det_intercept(Detector_1, p_sample_list,
                                                                   k_out_list, omega_list)

    # call display_detector for generating a diffraction image
    display_detector_bounded_animate(Detector_1, zeta_pix, new_omega, display_omega_bounds)

    return 0


# ************************************* Test Function Execution ************************************
test()
