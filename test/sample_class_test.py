# **************************************************************************************************
# Name:    sample_class_test.py
# Purpose: Test the sample classes from sample_class.py
# Input:   Listed below under "Variable Definitions" section
# Output:  none
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
# Standard Library
import numpy as np

# My Library
import classes.sample_class as sample_class
import utils.strain_functions as strain_func
import utils.math_functions as math_func


# *************************************** Variable Definitions *************************************

# unitCell_1 lattice parameters (a, b, c, alpha, beta, gamma) (Angstroms, degrees)
unitCell_1 = sample_class.UnitCell(np.array([2, 2, 2, 90, 90, 90]))

# Grain_1 parameters (unitCell, dimension, COM, orientation, intensity)
quat_1 = np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3])
Grain_1 = sample_class.Grain(unitCell_1, np.array([1.0, 1.0, 1.0]), np.array([0, 0, 0]), quat_1)

# Sample_1 parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
omegaLow = 0
omegaHigh = 180
Sample_1 = sample_class.Sample(np.array([Grain_1]), omegaLow, omegaHigh, 1)

# Mesh_1 parameters (grain, numX, numY, numZ)
Mesh_1 = sample_class.Mesh(Grain_1, 5, 5, 5)


# ************************************* Test Function Definition ***********************************
def test():
    planes = np.array([[1, 3, 4],
                       [1, 1, 1]])
    planes = math_func.normalize_rows(planes)
    strain = np.array([[.01],
                       [.02]])

    rose = strain_func.strain_rosette(strain, planes)


    return 0


# ************************************* Test Function Execution ************************************
test()
