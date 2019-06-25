# **************************************************************************************************
# Name:    sample_class.py
# Purpose: Module to provides sample classes for virtual diffraction
# Input:   none
# Output:  Module sample class definitions for virtual diffraction
# Notes:   none
# **************************************************************************************************


# ********************************************* Imports ********************************************
# Standard Library
import numpy as np
import scipy.constants as sciconst
# My Library


# ***************************************** Class Definitions **************************************

# Class: UnitCell
# Description: Class to hold crystal unit cell information
# Note: none
class UnitCell:
    # class variables
    latticeParams = np.zeros(6)  # (6x1 matrix) - lattice parameters in the form
                                 # [a,b,c,alpha,beta,gamma] (Angstroms, degrees)
    a1 = np.zeros(3)             # (3x1 vector) - unit cell vector 1
    a2 = np.zeros(3)             # (3x1 vector) - unit cell vector 2
    a3 = np.zeros(3)             # (3x1 vector) - unit cell vector 3
    volume = None                # (float) - unit cell volume

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
        #          matrix (2pi convention)
        # Input:   none
        # Output:  recip_mat (3x3 matrix) - columns of matrix are reciprocal lattice vectors
        # Notes:   none
        # ******************************************************************************************

        b1 = (2 * sciconst.pi / self.volume) * np.cross(self.a2, self.a3)
        b2 = (2 * sciconst.pi / self.volume) * np.cross(self.a3, self.a1)
        b3 = (2 * sciconst.pi / self.volume) * np.cross(self.a1, self.a2)
        return np.column_stack((b1, b2, b3))


# Class: Grain
# Description: Class to hold sample grain information
# Note: y is in the loading dir (positive = up), z is in the beam dir (positive = toward the source)
class Grain:
    # class variables
    unitCell = None                  # (UnitCell object) - holds all info on crystal unit cell
    grainDimensions = np.zeros(3)    # (3x1 vector) - grain dimensions (x,y,z) in mm
    grainCOM = np.zeros(3)           # (3x1 vector) - grain centroids/ center of mass (x,y,z) in
                                     # sample coord system in mm
    orientationQuat = np.zeros(4)    # (4x1 vector) - orientation of the grain, which provides
                                     # crystal to sample transformation (normalized quaternion)
                                     # (w, x, y, z)
    grainStrain = np.zeros(3)        # (3x3 matrix) - strain of the grain in tensor form in the
                                     # crystal coord system, zeros matrix is the initial value

    # Constructor
    def __init__(self, unit_cell_in, grain_dim_in, grain_com_in, orient_quat,
                 grain_strain_in=np.zeros(3)):
        self.unitCell = unit_cell_in
        self.grainDimensions = grain_dim_in
        self.grainCOM = grain_com_in
        self.orientationQuat = orient_quat
        self.grainStrain = grain_strain_in

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

        w = self.orientationQuat[0]
        x = self.orientationQuat[1]
        y = self.orientationQuat[2]
        z = self.orientationQuat[3]

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

        self.orientationQuat[0] = np.sqrt(1 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2
        self.orientationQuat[1] = (rot_mat[2, 1] - rot_mat[1, 2]) / (4 * self.orientationQuat[0])
        self.orientationQuat[2] = (rot_mat[0, 2] - rot_mat[2, 0]) / (4 * self.orientationQuat[0])
        self.orientationQuat[3] = (rot_mat[1, 0] - rot_mat[0, 1]) / (4 * self.orientationQuat[0])

    def vector2quat(self, orient_vec):
        # ******************************************************************************************
        # Name:    vector2quat
        # Purpose: function that sets orientation quaternion from an orientation vector in the
        #          sample coord system
        # Input:   orient_vec (3x1 vector) - orientation vector in the sample coord system, where
        #                                    the other vector is sample vector [0, 1, 0] (up in
        #                                    loading dir)
        # Output:  none
        # Notes:   none
        # ******************************************************************************************

        # initializing variables
        sample_vec = np.array([0, 1, 0])
        mag_orient = np.linalg.norm(orient_vec)
        mag_sample = np.linalg.norm(sample_vec)

        # cross product of these creates x, y, z of quaternion
        cross_vec = np.cross(sample_vec, orient_vec)

        # setting quaternion values
        self.orientationQuat[0] = (np.sqrt(mag_orient ** 2 * mag_sample ** 2)
                                   + np.dot(sample_vec, orient_vec))
        self.orientationQuat[1] = cross_vec[0]
        self.orientationQuat[2] = cross_vec[1]
        self.orientationQuat[3] = cross_vec[2]

        # normalize quaternion
        self.orientationQuat = self.orientationQuat / np.linalg.norm(self.orientationQuat)

    def dimen2meters(self):
        # ******************************************************************************************
        # Name:    dimen2meters
        # Purpose: function that returns the dimensions of the grain in meters
        # Input:   none
        # Output:  dimen_meters (3x1 vector) - holds the dimension of the grain in meters
        # Notes:   none
        # ******************************************************************************************

        return self.grainDimensions * 1e-3

    def reciprocal_strain(self):
        # ******************************************************************************************
        # Name:    reciprocal_strain
        # Purpose: function that returns reciprocal strain of the crystal stretched by (I-e) where
        #          e = grainStrain
        # Input:   none
        # Output:  recip_strain (3x3 matrix) - reciprocal strain of the crystal
        # Notes:   none
        # ******************************************************************************************

        return np.eye(3) - self.grainStrain


# Class: Mesh
# Description: Class to hold grain mesh information
# Note: none
class Mesh:
    # class variables
    grain = None       # (Grain object) - holds Grain object to hold grain info for mesh
    numBlocksX = None  # (float) - holds number of blocks in the x-dir of the mesh
    numBlocksY = None  # (float) - holds number of blocks in the y-dir of the mesh
    numBlocksZ = None  # (float) - holds number of blocks in the z-dir of the mesh
    sizeBlockX = None  # (float) - holds x-dimension of a block in the mesh (meters)
    sizeBlockY = None  # (float) - holds y-dimension of a block in the mesh (meters)
    sizeBlockZ = None  # (float) - holds z-dimension of a block in the mesh (meters)
    blockVol = None    # (float) - holds the volume of a block in the mesh (m^3)
    mesh = None        # (p x 4 matrix) - holds the mesh position (m) and volume (m^3) in the form
                       # [x, y, z, vol] for p mesh blocks where p = X * Y * Z numblocks

    # Constructor
    def __init__(self, grain_in, blocks_x_in=1, blocks_y_in=1, blocks_z_in=1):
        self.grain = grain_in
        self.numBlocksX = blocks_x_in
        self.numBlocksY = blocks_y_in
        self.numBlocksZ = blocks_z_in
        self.sizeBlockX = self.grain.grainDimensions[0] / self.numBlocksX
        self.sizeBlockY = self.grain.grainDimensions[1] / self.numBlocksY
        self.sizeBlockZ = self.grain.grainDimensions[2] / self.numBlocksZ
        self.blockVol = self.sizeBlockX * self.sizeBlockY * self.sizeBlockZ
        self.generate_mesh()

    # Other Functions
    def generate_mesh(self):
        # ******************************************************************************************
        # Name:    generate_mesh
        # Purpose: function that generates the mesh for this object in the form of parallelopiped
        # Input:   none
        # Output:  none
        # Notes:   Dimension units in mm
        # ******************************************************************************************

        # initialize variables
        total_blocks = self.numBlocksX * self.numBlocksY * self.numBlocksZ

        # create array of positions for mesh blocks centered at (0,0,0)
        block_x = np.linspace(-self.grain.grainDimensions[0] / 2 + self.sizeBlockX / 2,
                              self.grain.grainDimensions[0] / 2 - self.sizeBlockX / 2,  self.numBlocksX)
        block_y = np.linspace(-self.grain.grainDimensions[1] / 2 + self.sizeBlockY / 2,
                              self.grain.grainDimensions[1] / 2 - self.sizeBlockY / 2, self.numBlocksY)
        block_z = np.linspace(-self.grain.grainDimensions[2] / 2 + self.sizeBlockZ / 2,
                              self.grain.grainDimensions[2] / 2 - self.sizeBlockZ / 2, self.numBlocksZ)

        # generate mesh of positions
        mesh = np.array(np.meshgrid(block_x, block_y, block_z)).T.reshape(-1, 3)

        # add column of volumes
        volumes = np.ones(total_blocks).reshape((total_blocks, 1))
        volumes = volumes * self.blockVol
        mesh = np.hstack((mesh, volumes))

        # add to object mesh
        self.mesh = mesh


# Class: Sample
# Description: Class to hold sample information
# Note: none
class Sample:
    # class variables
    grains = np.empty(1, dtype=Grain)  # (n x 1 vector) - holds n grains in sample
    meshes = np.empty(1, dtype=Mesh)   # (n x 1 vector) - holds n mesh in sample
    numGrains = None                   # (integer) - number of grains in sample
    omegaLow = None                    # (float) - lower bound of sample rotation omega (degrees)
    omegaHigh = None                   # (float) - upper bound of sample rotation omega (degrees)
    omegaStepSize = None               # (float) - step size of sample rotation omega (degrees)

    # Constructor
    def __init__(self, grains_in, omega_low_in, omega_high_in, omega_step_size_in, meshes_in=-1):
        self.grains = grains_in
        self.numGrains = self.grains.shape[0]
        self.omegaLow = omega_low_in
        self.omegaHigh = omega_high_in
        self.omegaStepSize = omega_step_size_in
        self.meshes = meshes_in


