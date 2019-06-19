# **************************************************************************************************
# Name:    sample_class_test.py
# Purpose: Test the sample classes from sample_class.py
# Input:   Listed below under "Variable Definitions" section
# Output:  none
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
from classes.sample_class import UnitCell, Grain, Sample, Mesh
from utils.sample_functions import read_grains_from_csv
from utils.math_functions import normalize
import numpy as np
import os

# *************************************** Variable Definitions *************************************

# unitCell_1 lattice parameters (a, b, c, alpha, beta, gamma) (Angstroms, degrees)
unitCell_1 = UnitCell(np.array([2, 2, 2, 90, 90, 90]))

# Grain_1 parameters (unitCell, dimension, COM, orientation, intensity)
quat_1 = np.array([7.356e-1, 6.616e-1, 1.455e-1, -8.024e-3])
Grain_1 = Grain(unitCell_1, np.array([1.0, 1.0, 1.0]), np.array([0, 0, 0]), quat_1)

# Sample_1 parameters (grains in a list, omegaLow, omegaHigh, omegaStepSize) (degrees)
omegaLow = 0
omegaHigh = 180
Sample_1 = Sample(np.array([Grain_1]), omegaLow, omegaHigh, 1)

# Mesh_1 parameters (grain, numX, numY, numZ)
Mesh_1 = Mesh(Grain_1, 5, 5, 5)


# ************************************* Test Function Definition ***********************************
def test():
    Grain_1.vector2quat(np.array([0, 0, 1]))
    print(Grain_1.orientationQuat)

    path = os.getcwd()
    path = path = path.split("test")[0] + "data\\multi_grains_15.csv"  # windows
    grain_list = read_grains_from_csv(path)


    return 0


# ************************************* Test Function Execution ************************************
test()
