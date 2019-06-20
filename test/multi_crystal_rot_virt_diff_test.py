# **************************************************************************************************
# Name:    multi_crystal_rot_virt_diff_test.py
# Purpose: Test the multi_cyrstal functions in virtual_diffractometer_functions.py
# Input:   Listed below under "Variable Definitions" section
# Output:  Diffraction image
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
# Standard Library
import numpy as np
import time
# My Library
import classes.equipment_class as equip_class
import classes.sample_class as sample_class
import utils.sample_functions as sample_func
import utils.virtual_diffractometer_functions as virt_diff_func
import utils.io_functions as io_func

# *************************************** Variable Definitions *************************************

# Detector parameters (distance, width, height, pixel_density)(mm, pixel, pixel, pixel/mm)
detector_size = 20000
Detector_1 = equip_class.Detector(1000, detector_size, detector_size, 10)

# LabSource parameters (energy, incomingXray) (keV, unitVector)
LabSource_1 = equip_class.LabSource(55.618, equip_class.LabSource.lab_z)


# ************************************* Test Function Definition ***********************************
def test():
    # initialize grain and mesh list
    mesh_size = 1
    grain_list = io_func.read_grains_from_csv("multi_grains_15.csv")
    mesh_list = sample_func.mesh_list_from_grain_list(grain_list, mesh_size)

    # Sample parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
    omegaLow = 0
    omegaHigh = 180
    omegaStepSize = 1
    Sample_1 = sample_class.Sample(np.array(grain_list), omegaLow, omegaHigh, omegaStepSize,
                                   np.array(mesh_list))

    # initialize hkl vectors and omega_bounds
    hkl_list = io_func.read_hkl_from_csv("hkl_list_10.csv")
    omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]
    display_omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]

    print("Starting #1")
    # call multi_crystal_diff, time is for measuring length of calculation
    t = time.time()
    multi_rot_diff_list = virt_diff_func.multi_crystal_rot_diff_exp(LabSource_1, Sample_1.grains,
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

    # call multi-crystal detector intercept
    [zeta, zeta_pix, new_omega] = virt_diff_func.multi_crystal_find_det_intercept(Detector_1,
                                                                                  p_sample_list,
                                                                                  k_out_list,
                                                                                  omega_list)
    # call display_detector for generating a diffraction image
    virt_diff_func.display_detector_bounded(Detector_1, zeta_pix, new_omega, display_omega_bounds)

    # call display_detector for generating a diffraction animation
    virt_diff_func.display_detector_bounded_animate(Detector_1, zeta_pix, new_omega,
                                                    display_omega_bounds)

    return 0


# ************************************* Test Function Execution ************************************
test()
