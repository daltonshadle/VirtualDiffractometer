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

