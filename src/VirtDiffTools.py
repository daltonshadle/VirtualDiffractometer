# **************************************************************************************************
# Name:    VirtDiffTools.py
# Purpose: Module to provides classes and tools for virtual diffraction
# Input:   none
# Output:  Module class definitions for virtual diffraction
# Notes:   none
# **************************************************************************************************


# ********************************************* Imports ********************************************
import numpy as np
import scipy.constants as sciconst
import matplotlib.pyplot as plot


# ***************************************** Class Definitions **************************************
# Class: LabSource
# Description: Class to hold lab source information
# Note: none
class LabSource:
    # class variables
    beamEnergy = None            # Energy of beam (keV)
    k_in_lab = np.zeros(3)       # Incoming wave vector in lab frame (Angstroms)
    lab_x = np.array([1, 0, 0])  # Lab x-axis unit vector
    lab_y = np.array([0, 1, 0])  # Lab y-axis unit vector
    lab_z = np.array([0, 0, 1])  # Lab z-axis unit vector

    # Constructor
    def __init__(self, energy_In, incoming_In):
        self.beamEnergy = energy_In
        self.k_in_lab = ((2 * sciconst.pi) / self.kev_2_angstroms()) * (-1 * incoming_In)

    # Other Functions
    def kev_2_angstroms(self):
        # ******************************************************************************************
        # Name:    kev_2_angstroms
        # Purpose: function that returns the wavelength of the beam energy in Angstroms
        # Input:   none
        # Output:  wavelength (float) - wavelength of the beam energy photons
        # Notes:   none
        # ******************************************************************************************

        return (sciconst.h * sciconst.c * 1e10) / (1000 * self.beamEnergy * sciconst.e)  # Angstroms

    def kev_2_meters(self):
        # ******************************************************************************************
        # Name:    kev_2_meters
        # Purpose: function that returns the wavelength of the beam energy in meters
        # Input:   none
        # Output:  wavelength (float) - wavelength of the beam energy photons
        # Notes:   none
        # ******************************************************************************************

        return (sciconst.h * sciconst.c) / (1000 * self.beamEnergy * sciconst.e)  # meters


# Class: Detector
# Description: Class to hold detector information
# Note: none
class Detector:
    # class variables
    distance = None    # Distance from detector to sample (mm)
    width = None       # Width of detector (pixel*e-4)
    height = None      # Height of detector (pixel*e-4)

    # Constructor
    def __init__(self, dist_In, width_In, height_In):
        self.distance = dist_In
        self.width = width_In
        self.height = height_In

    # Other Functions
    def dist_2_meters(self):
        # ******************************************************************************************
        # Name:    dist_2_meters
        # Purpose: function that returns the sample to detector distance in meters
        # Input:   none
        # Output:  distance (float) - distance from sample to detector
        # Notes:   none
        # ******************************************************************************************

        return self.distance * 1E-3  # meters


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
    def __init__(self, latticeParams_In):
        # Round precision variable for lattice vectors
        precision = 15

        self.latticeParams = latticeParams_In
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
    def __init__(self, unitCell_In, grainDim_In, grainCOM_In, orient_In, intensity_In):
        self.unitCell = unitCell_In
        self.grainDimensions = grainDim_In
        self.grainCOMs = grainCOM_In
        self.orientation = orient_In
        self.intensity = intensity_In

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
    def __init__(self, grains_In, omegaLow_In, omegaHigh_In, omegaStepSize_In):
        self.grains = grains_In
        self.numGrains = self.grains.shape[0]
        self.omegaLow = omegaLow_In
        self.omegaHigh = omegaHigh_In
        self.omegaStepSize = omegaStepSize_In

