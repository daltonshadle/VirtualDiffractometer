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
from utils.virtual_diffractometer_functions import rotating_diffraction_experiment, \
    find_detector_intercept, display_detector_intercept
import numpy as np

# *************************************** Variable Definitions *************************************

# Detector_1 parameters (distance, energy, width, height)(mm, keV, pixel*e-4, pixel*e-4)
detector_size = 500
Detector_1 = Detector(1000, detector_size, detector_size, 1)

# LabSource_1 parameters (energy, incomingXray) (keV, unitVector)
LabSource_1 = LabSource(55.618, LabSource.lab_z)

# unitCell_1 lattice parameters (a, b, c, alpha, beta, gamma) (Angstroms, degrees)
unitCell_1 = UnitCell(np.array([2, 2, 2, 90, 90, 90]))

# Grain_1 parameters (unitCell, dimension, COM, orientation, intensity)
Grain_1 = Grain(unitCell_1, np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.5, 0.0]),
                np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3]), 100)
Grain_1.rotmat2quat(np.eye(3))

# Sample_1 parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
Sample_1 = Sample(np.array([Grain_1]), 0, 180, 1)


# ************************************* Test Function Definition ***********************************
def test():
    path = "C:\Git Repositories\VirtualDiffractometer\data\hkl_list_1.csv"
    hkl_list = read_hkl_from_csv(path)
    hkl_list = gen_hkl_fam_from_list(hkl_list, cubic=True)

    omega_tuple = [Sample_1.omegaLow, Sample_1.omegaHigh, Sample_1.omegaStepSize]
    [two_theta, eta, k_out_lab, omega] = rotating_diffraction_experiment(LabSource_1,
                                                                         Sample_1.grains[0],
                                                                         hkl_list, omega_tuple)

    p_lab = np.transpose(np.array([[0, 0, 0], ] * np.shape(k_out_lab)[1]))
    [zeta, zeta_pix] = find_detector_intercept(Detector_1, k_out_lab, p_lab)

    display_omega_bounds = [0, 180]
    display_detector_intercept(Detector_1, zeta_pix, omega, display_omega_bounds)

    return 0


# ************************************* Test Function Execution ************************************
test()
