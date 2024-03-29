# **************************************************************************************************
# Name:    sample_functions.py
# Purpose: Module to provides sample related functions for virtual diffraction
# Input:   none
# Output:  Module for sample related function definitions for virtual diffraction
# Notes:   none
# **************************************************************************************************


# ********************************************* Imports ********************************************
# Standard Library
import numpy as np
# My Library
import classes.sample_class as sample_class


# *************************************** Function Definitions *************************************
def gen_hkl_fam(hkl_vec, cubic=False, hexagonal=False):
    # **********************************************************************************************
    # Name:    gen_hkl_fam
    # Purpose: function that generates the hkl family plane vectors based on the given input for
    #          either cubic or hexagonal
    # Input:   hkl_vec (1x3 vector) - used to define the hkl family
    #          cubic (boolean) - used to declare cubic symmetry (true = use)
    #          hexagonal (boolean) - used to declare hexagonal symmetry (true = use)
    # Output:  hkl_fam_list (n x 3 matrix) - matrix that holds all the hkl family vectors
    # Notes:   none
    # **********************************************************************************************

    # Initialize variables
    hkl_fam_list = hkl_vec
    h = hkl_vec[0]
    k = hkl_vec[1]
    l = hkl_vec[2]

    # Start computation based on booleans
    if cubic:
        # Evaluating Cubic, initialize starting values
        check = np.abs(h) + np.abs(k) + np.abs(l)
        values = np.array([h, -h, k, -k, l, -l])

        # Iterate over the range of values and calc possible hkl vectors, add to list if in fam
        for m in range(0, len(values)):
            for n in range(0, len(values)):
                for o in range(0, len(values)):
                    test = np.abs(values[m]) + np.abs(values[n]) + np.abs(values[o])
                    if test == check:
                        temp_hkl = np.array([values[m], values[n], values[o]])
                        hkl_fam_list = np.vstack((hkl_fam_list, temp_hkl))
    elif hexagonal:
        # Evaluating Cubic, initialize starting values
        i = -(h + k)
        check = np.abs(h) + np.abs(k) + np.abs(i)
        values = np.array([h, -h, k, -k, i, -i])

        # Iterate over the range of values and calc possible hkl vectors, add to list if in fam
        for m in range(0, len(values)):
            for n in range(0, len(values)):
                for o in range(0, len(values)):
                    test = np.abs(values[m]) + np.abs(values[n]) + np.abs(values[o])
                    if test == check:
                        if (values[m] + values[n]) == -values[o]:
                            temp_hkl = np.array([values[m], values[n], values[o]])
                            hkl_fam_list = np.vstack((hkl_fam_list, temp_hkl))

        # Remove i column
        hkl_fam_list = hkl_fam_list[:, [0, 2]]
        l_column = np.ones((np.shape(hkl_fam_list)[0], 1))

        # Re-add l column to fam list
        positive_l = np.hstack((hkl_fam_list, l_column))
        negative_l = np.hstack((hkl_fam_list, -1 * l_column))
        hkl_fam_list = np.vstack((positive_l, negative_l))
    else:
        # No choice selected, report error
        print("gen_hkl_fam_list: No choice selected, returning empty hkl_fam_list")

    hkl_fam_list = np.unique(hkl_fam_list, axis=0)
    return hkl_fam_list


def gen_hkl_fam_from_list(hkl_list, cubic=False, hexagonal=False):
    # **********************************************************************************************
    # Name:    gen_hkl_fam_from_list
    # Purpose: function that generates the hkl family plane vectors based on the given input for
    #          either cubic or hexagonal
    # Input:   hkl_list (n x 3 matrix) - list of hkl plane vectors to find fams
    #          cubic (boolean) - used to declare cubic symmetry (true = use)
    #          hexagonal (boolean) - used to declare hexagonal symmetry (true = use)
    # Output:  hkl_fam_list (m x 3 matrix) - matrix that holds all the hkl family vectors
    # Notes:   none
    # **********************************************************************************************

    # initialize return list
    hkl_fam_list = np.empty((0, 3))

    # iterate over each hkl vector in list and find the families of that vector, add to total list
    for i in hkl_list:
        temp_list = gen_hkl_fam(np.squeeze(np.asarray(i)), cubic=cubic, hexagonal=hexagonal)
        hkl_fam_list = np.vstack((hkl_fam_list, temp_list))

    hkl_fam_list = np.unique(hkl_fam_list, axis=0)
    return hkl_fam_list


def mesh_list_from_grain_list(grain_list, mesh_size):
    # **********************************************************************************************
    # Name:    mesh_list_from_grain_list
    # Purpose: function that creates a mesh list from a grain list
    # Input:   grain_list (list) - list of grains
    #          mesh_size (float) - size of mesh used
    # Output:  mesh_list (list) - list of meshes created from grains
    # Notes:   none
    # **********************************************************************************************

    # initialize return list
    mesh_list = []

    for item in grain_list:
        mesh_list.append(sample_class.Mesh(item, mesh_size, mesh_size, mesh_size))

    return mesh_list


def find_hkl_from_g_sample(g_sample, grain):
    # **********************************************************************************************
    # Name:    find_hkl_from_g_sample
    # Purpose: function computes hkl list from reciprocal lattice vectors in the sample coord system
    #          by solving Ax=b
    # Input:   g_sample (3 x m) - m reciprocal lattice vectors in the sample coord system
    #          grain (obejct) - grain object for grain information
    # Output:  hkl_list (3 x m matrix) - matrix that holds m the hkl family vectors
    # Notes:   none
    # **********************************************************************************************

    # initialize A matrix, take inverse
    mat_a = (grain.quat2rotmat() * grain.reciprocal_strain()
             * grain.unitCell.get_reciprocal_lattice_vectors())
    mat_a = np.linalg.pinv(mat_a)

    # solve for x, hkl_list
    hkl_list = mat_a * g_sample

    return hkl_list


def match_g_index(g_index1, g_index2):
    # **********************************************************************************************
    # Name:    match_g_index
    # Purpose: function matches g_index values for producing similar spots
    # Input:   g_index1 (m x 1) - m indices of the g_sample vectors used in rotating diffraction exp
    #          g_index2 (n x 1) - n indices of the g_sample vectors used in rotating diffraction exp
    # Output:  index1 (p x 1) - indices for matching g_index in sample 1
    #          index2 (p x 1) - indices for matching g_index in sample 2
    # Notes:   none
    # **********************************************************************************************

    # determine indices where both match one another
    temp_index1 = np.in1d(g_index1, g_index2)
    temp_index2 = np.in1d(g_index2, g_index1)

    # create array of indices the size of both g_index
    index1 = np.array(range(g_index1.shape[0]))
    index2 = np.array(range(g_index2.shape[0]))

    # keep the indices that are shared in both sets
    index1 = index1[temp_index1]
    index2 = index2[temp_index2]

    return index1, index2


def structure_factor(hkl_list, unit_cell):
    # **********************************************************************************************
    # Name:    structure_factor
    # Purpose: function determines the structure factor based on hkl and unit cell
    # Input:   hkl_list (3 x m) - m hkl planes for processing
    #          unit_cell (object) - unit cell object for processing
    # Output:  structure_factor (m x 1) - structure factor list of unit cell
    # Notes:   none
    # **********************************************************************************************

    return 0


def create_fcc_hkl_list(hkl_int):
    # **********************************************************************************************
    # Name:    create_fcc_hkl_list
    # Purpose: function create the diffraction hkl list for fcc crystals
    # Input:   hkl_int (int) - highest Miller index
    # Output:  fcc_hkl_list (3 x m) - hkl list for diffracting fcc crystals
    # Notes:   none
    # **********************************************************************************************

    # initialize return variable
    fcc_hkl_list = np.zeros((1, 3))

    # iterate over all possible hkl combinations
    for m in range(-hkl_int, hkl_int + 1, 1):
        for n in range(-hkl_int, hkl_int + 1, 1):
            for o in range(-hkl_int, hkl_int + 1, 1):
                # check if hkl combination meets fcc criteria (all odd or all even)
                if ((m%2 == 0 and n%2 == 0 and o%2 == 0) or (m%2 == 1 and n%2 == 1 and o%2 == 1)) \
                        and (abs(m) + abs(n) + abs(o)) != 0:
                    temp_hkl = np.array([m, n, o])
                    fcc_hkl_list = np.vstack((fcc_hkl_list, temp_hkl))

    # return value, remove first row of zeroes
    fcc_hkl_list = np.delete(fcc_hkl_list, 0, axis=0)
    return fcc_hkl_list


def create_bcc_hkl_list(hkl_int):
    # **********************************************************************************************
    # Name:    create_bcc_hkl_list
    # Purpose: function create the diffraction hkl list for bcc crystals
    # Input:   hkl_int (int) - highest Miller index
    # Output:  bcc_hkl_list (3 x m) - hkl list for diffracting bcc crystals
    # Notes:   none
    # **********************************************************************************************

    # initialize return variable
    bcc_hkl_list = np.zeros((1, 3))

    # iterate over all possible hkl combinations
    for m in range(-hkl_int, hkl_int + 1, 1):
        for n in range(-hkl_int, hkl_int + 1, 1):
            for o in range(-hkl_int, hkl_int + 1, 1):
                # check if hkl combination meets fcc criteria (h+k+l = even number)
                if (m + n + o)%2 == 0 and (abs(m) + abs(n) + abs(o)) != 0:
                    temp_hkl = np.array([m, n, o])
                    bcc_hkl_list = np.vstack((bcc_hkl_list, temp_hkl))

    # return value, remove first row of zeroes
    bcc_hkl_list = np.delete(bcc_hkl_list, 0, axis=0)
    return bcc_hkl_list

def create_grain_list():
    # **********************************************************************************************
    # Name:    create_grain_list
    # Purpose: function creates grain list from parameters for a sample
    # Input:   hkl_int (int) - highest Miller index
    # Output:  bcc_hkl_list (3 x m) - hkl list for diffracting bcc crystals
    # Notes:   none
    # **********************************************************************************************

    return 0



