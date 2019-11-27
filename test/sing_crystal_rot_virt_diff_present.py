# **************************************************************************************************
# Name:    sing_crystal_rot_virt_diff_test.py
# Purpose: Test the single_crystal functions in virtual_diffractometer_functions.py
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
import utils.math_functions as math_func
import utils.sample_functions as sample_func
import utils.strain_functions as strain_func
import utils.virtual_diffractometer_functions as virt_diff_func
import utils.io_functions as io_func


# *************************************** Variable Definitions *************************************

# Detector_1 parameters (distance, width, height, pixel_density)(mm, pixel, pixel, pixel/mm)
detector_size = 20000
Detector_1 = equip_class.Detector(1000, detector_size, detector_size, 10)

# LabSource_1 parameters (energy, incomingXray) (keV, unitVector)
LabSource_1 = equip_class.LabSource(61.332, equip_class.LabSource.lab_z)

# unitCell_1 lattice parameters (a, b, c, alpha, beta, gamma) (Angstroms, degrees)
unitCell_1 = sample_class.UnitCell(np.array([1, 1, 10, 90, 90, 90]))

# Grain_1 parameters (unitCell, dimension, COM, orientation)
quat_1 = np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3])
vec_1 = math_func.normalize(np.array([.3, .7, .4]))
Grain_1 = sample_class.Grain(unitCell_1, np.array([1.0, 1.0, 1.0]), np.array([0, 0, 0]), quat_1)
Grain_1.vector2quat(vec_1)

# Mesh_1 parameters (grain, numX, numY, numZ)
Mesh_1 = sample_class.Mesh(Grain_1, 1, 1, 1)

# Sample_1 parameters (grains_list, omegaLow, omegaHigh, omegaStepSize, meshes_list) (degrees)
omegaStepSize = 5
omegaLow = 2.5 - omegaStepSize/2
omegaHigh = 177.5 + omegaStepSize/2
Sample_1 = sample_class.Sample(np.array([Grain_1]), omegaLow, omegaHigh, omegaStepSize,
                               np.array([Mesh_1]))

# Display options
process_1 = True
process_2 = True
process_3 = False
process_4 = False
process_5 = False


# ************************************* Test Function Definition ***********************************
def test():
    # initialize hkl vectors and omega_bounds
    hkl_list = sample_func.create_fcc_hkl_list(3)
    omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]
    display_omega_bounds = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]

    # GRAIN WITHOUT STRAIN -------------------------------------------------------------------------
    # start process #1
    if process_1:
        t = time.time()
        # call single crystal rotating diffraction experiment, time is for measuring calculation
        # length
        [two_theta1, eta, k_out_lab, omega1, g_sample1, g_index1] = \
            virt_diff_func.sing_crystal_rot_diff_exp(LabSource_1, Sample_1.grains[0], hkl_list,
                                                     omega_bounds)
        print("Process #1 Elapsed: ", time.time() - t)

    # start process #2
    if process_2:
        t = time.time()
        # call single crystal intercept mesh and display detector to display a diffraction image
        [zeta, zeta_pix, new_k_out, new_omega] = \
            virt_diff_func.sing_crystal_find_det_intercept_mesh(Detector_1, Sample_1.meshes[0],
                                                                k_out_lab, omega1)
        virt_diff_func.display_detector_bounded(Detector_1, zeta_pix, new_omega, display_omega_bounds)
        print("Process #2 Elapsed: ", time.time() - t)

    # GRAIN WITH STRAIN ----------------------------------------------------------------------------
    # initialize strain value
    Sample_1.grains[0].grainStrain = np.array([[.001,  0,   -.001],
                                               [0,  .001,   0],
                                               [-.001,  0,   .0022]])

    # start process #3
    if process_3:
        t = time.time()
        # call single crystal rotating diffraction experiment, time is for measuring calculation
        # length
        [two_theta2, eta, k_out_lab, omega2, g_sample2, g_index2] = \
            virt_diff_func.sing_crystal_rot_diff_exp(LabSource_1, Sample_1.grains[0], hkl_list,
                                                     omega_bounds)
        print("Process #3 Elapsed: ", time.time() - t)

    # start process #4
    if process_4:
        t = time.time()
        # call single crystal intercept mesh and display detector to display a diffraction image
        [zeta, zeta_pix, new_k_out, new_omega] = \
            virt_diff_func.sing_crystal_find_det_intercept_mesh(Detector_1, Sample_1.meshes[0],
                                                                k_out_lab, omega2)
        virt_diff_func.display_detector_bounded(Detector_1, zeta_pix, new_omega, display_omega_bounds)
        print("Process #4 Elapsed: ", time.time() - t)

    # start process #5
    if process_5:
        t = time.time()
        # call single crystal intercept and display detector to display a diffraction animation
        [zeta, zeta_pix] = virt_diff_func.sing_crystal_find_det_intercept(Detector_1,
                                                                          Sample_1.grains[0].grainCOM,
                                                                          k_out_lab, omega2)
        virt_diff_func.display_detector_bounded_animate(Detector_1, zeta_pix, omega2,
                                                        display_omega_bounds)
        print("Process #5 Elapsed: ", time.time() - t)

    # STRAIN CALCULATION ---------------------------------------------------------------------------
    # matching g_index values and reducing two_theta and g_sample data
    index1, index2 = sample_func.match_g_index(g_index1, g_index2)

    two_theta1 = two_theta1[index1]
    g_sample1 = g_sample1[:, index1]
    two_theta2 = two_theta2[index2]
    g_sample2 = g_sample2[:, index2]

    strain_vec = strain_func.calc_lattice_strain_from_two_theta(np.transpose(g_sample1),
                                                                np.transpose(two_theta1),
                                                                np.transpose(two_theta2))

    print("Input Strain Tensor (Crystal Coord): \n", Sample_1.grains[0].grainStrain)
    print("Output Strain Tensor (Sample Coord): \n", strain_func.strain_vec2tensor(strain_vec))

    return 0


# ************************************* Test Function Execution ************************************
test()
