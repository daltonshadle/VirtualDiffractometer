# **************************************************************************************************
# Name:    sample_class.py
# Purpose: Module to provides sample classes for virtual diffraction
# Input:   none
# Output:  Module sample class definitions for virtual diffraction
# Notes:   none
# **************************************************************************************************


# ********************************************* Imports ********************************************
import numpy as np


# ***************************************** Class Definitions **************************************

# Class: UnitCell
# Description: Class to hold crystal unit cell information
# Note: none
class UnitCell:
    # class variables
    latticeParams = np.zeros(6)  # lattice parameters in the form [a,b,c,alpha,beta,gamma]
                                 # (Angstroms, degrees)
    a1 = np.zeros(3)             # unit cell vector 1
    a2 = np.zeros(3)             # unit cell vector 2
    a3 = np.zeros(3)             # unit cell vector 3
    volume = None                # unit cell volume

    # Constructor
    def __init__(self, lattice_params_in):
        # Round precision variable for lattice vectors
        precision = 15

        self.latticeParams = lattice_params_in
        a = self.latticeParams[0]
        b = self.latticeParams[1]
        c = self.latticeParams[2]
        alpha = self.latticeParams[3]
        beta = self.latticeParams[4]
        gamma = self.latticeParams[5]

        cx = c * np.cos(np.deg2rad(beta))
        cy = (c * (np.cos(np.deg2rad(alpha)) - np.cos(np.deg2rad(beta)) * np.cos(np.deg2rad(gamma)))
             / np.sin(np.deg2rad(gamma)))
        cz = np.sqrt(c**2 - cx**2 - cy**2)

        self.a1 = np.round(a * np.array([1, 0, 0]), precision)
        self.a2 = np.round(b * np.array([np.cos(np.deg2rad(gamma)), np.sin(np.deg2rad(gamma)), 0]),
                           precision)
        self.a3 = np.round(np.array([cx, cy, cz]), precision)
        self.volume = np.dot(self.a1, np.cross(self.a2, self.a3))

    def get_reciprocal_lattice_vectors(self):
        # ******************************************************************************************
        # Name:    get_reciprocal_lattice_vectors
        # Purpose: function that returns the reciprocal lattice vectors as the columns of a 3x3
        #          matrix
        # Input:   none
        # Output:  recip_mat (3x3 matrix) - columns of matrix are reciprocal lattice vectors
        # Notes:   none
        # ******************************************************************************************

        b1 = (2 / self.volume) * np.cross(self.a2, self.a3)
        b2 = (2 / self.volume) * np.cross(self.a3, self.a1)
        b3 = (2 / self.volume) * np.cross(self.a1, self.a2)
        return np.column_stack((b1, b2, b3))


# Class: Grain
# Description: Class to hold sample grain information
# Note: y is in the loading dir (positive = up), z is in the beam dir (positive = toward the source)
class Grain:
    # class variables
    unitCell = None                # holds all info on crystal unit cell
    grainDimensions = np.zeros(3)  # grain dimensions (x,y,z)
    grainCOMs = np.zeros(3)        # grain centroids/ center of mass (x,y,z)
    orientation = np.zeros(4)      # orientation of the grain, which provides crystal to sample
                                   # transformation (quaternions)
    intensity = None               # Set intensities for all spots of each grain (nice way to
                                   # differentiate by sight)

    # Constructor
    def __init__(self, unit_cell_in, grain_dim_in, grain_com_in, orient_in, intensity_in):
        self.unitCell = unit_cell_in
        self.grainDimensions = grain_dim_in
        self.grainCOMs = grain_com_in
        self.orientation = orient_in
        self.intensity = intensity_in

    # Other Functions
    def quat2rotmat(self):
        # ******************************************************************************************
        # Name:    quat2rotmat
        # Purpose: function that returns the rotation matrix from the orientation
        #          quaternion (crystal to sample)
        # Input:   none
        # Output:  rot_mat (3x3 matrix) - rotation matrix for grain
        # Notes:   none
        # ******************************************************************************************

        w = self.orientation[0]
        x = self.orientation[1]
        y = self.orientation[2]
        z = self.orientation[3]

        rot_mat = np.matrix([[1-2*(y**2-z**2), 2*(x*y-w*z),     2*(x*z+w*y)],
                             [2*(x*y+w*z),     1-2*(x**2-z**2), 2*(y*z-w*x)],
                             [2*(x*z-w*y),     2*(y*z+w*x),     1-2*(x**2-y**2)]])

        return rot_mat

    def rotmat2quat(self, rot_mat):
        # ******************************************************************************************
        # Name:    rotmat2quat
        # Purpose: function that sets orientation quaternion from rotation matrix (crystal to
        #          sample)
        # Input:   rot_mat (3x3 matrix) - rotation matrix for grain
        # Output:  none
        # Notes:   none
        # ******************************************************************************************

        self.orientation[0] = np.sqrt(1 + rot_mat[0][0] + rot_mat[1][1] + rot_mat[2][2]) / 2
        self.orientation[1] = (rot_mat[2][1] - rot_mat[1][2]) / (4 * self.orientation[0])
        self.orientation[2] = (rot_mat[0][2] - rot_mat[2][0]) / (4 * self.orientation[0])
        self.orientation[3] = (rot_mat[1][0] - rot_mat[0][1]) / (4 * self.orientation[0])


# Class: Sample
# Description: Class to hold sample information
# Note: none
class Sample:
    # class variables
    grains = np.array((1, 1), dtype=Grain)  # holds all grains in sample,
                                            # ENTERED AS VERTICAL VECTOR OF GRAINS
    numGrains = None                        # number of grains in sample
    omegaLow = None                         # lower bound of sample rotation omega (degrees)
    omegaHigh = None                        # upper bound of sample rotation omega (degrees)
    omegaStepSize = None                    # step size of sample rotation omega (degrees)

    # Constructor
    def __init__(self, grains_in, omega_low_in, omega_high_in, omega_step_size_in):
        self.grains = grains_in
        self.numGrains = self.grains.shape[0]
        self.omegaLow = omega_low_in
        self.omegaHigh = omega_high_in
        self.omegaStepSize = omega_step_size_in

