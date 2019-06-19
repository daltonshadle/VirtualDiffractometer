# **************************************************************************************************
# Name:    sample_functions.py
# Purpose: Module to provides sample related functions for virtual diffraction
# Input:   none
# Output:  Module for sample related function definitions for virtual diffraction
# Notes:   none
# **************************************************************************************************


# ********************************************* Imports ********************************************
import numpy as np
import csv
from classes.sample_class import UnitCell, Grain, Mesh
from utils.math_functions import normalize


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

        # Iterate over the range of values and compute possible hkl vectors, add to list if in fam
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

        # Iterate over the range of values and compute possible hkl vectors, add to list if in fam
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


def read_hkl_from_csv(file_path):
    # **********************************************************************************************
    # Name:    read_hkl_from_csv
    # Purpose: function that reads the hkl vector list from csv and returns as list
    # Input:   file_path (string) - path of csv file to read from, csv must store hkl as (n x 3)
    # Output:  hkl_list (n x 3 matrix) - matrix that holds all the hkl vectors from csv file
    # Notes:   none
    # **********************************************************************************************

    # initialize return list
    hkl_list = np.empty((0, 3))

    # open csv storing hkl vectors
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        # iterate over all rows in csv and add to hkl_list
        for row in csv_reader:
            hkl_vec = np.array([float(row[0]), float(row[1]), float(row[2])])
            hkl_list = np.vstack((hkl_list, hkl_vec))
            line_count += 1
        print(f'Processed {line_count} hkl vectors.')
    return hkl_list


def read_grains_from_csv(file_path):
    # **********************************************************************************************
    # Name:    read_grains_from_csv
    # Purpose: function that reads grain data from csv and returns as list of grains
    # Input:   file_path (string) - path of csv file to read from, csv must store grains as (n x 13)
    # Output:  grain_list (list) - matrix that holds n grains from csv file
    # Notes:   none
    # **********************************************************************************************

    # initialize return list
    grain_list = []

    # open csv storing hkl vectors
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        temp_cell = UnitCell(np.array([4, 4, 4, 90, 90, 90]))

        # iterate over all rows in csv and add to grain list
        for row in csv_reader:
            if line_count != 0:
                dim_vec = np.array([float(row[0]), float(row[1]), float(row[2])])
                com_vec = np.array([float(row[3]), float(row[4]), float(row[5])])
                quat_vec = np.array([float(row[6]), float(row[7]), float(row[8]), float(row[9])])
                orient_vec = np.array([float(row[10]), float(row[11]), float(row[1])])

                # normalize vectors
                quat_vec = normalize(quat_vec)
                orient_vec = normalize(orient_vec)

                # Grain (unitCell, dimension, COM, orientation)
                temp_grain = Grain(temp_cell, dim_vec, com_vec, quat_vec)
                temp_grain.vector2quat(orient_vec)

                # add to list
                grain_list.append(temp_grain)
            line_count += 1
        print(f'Processed {line_count} grains.')
    return grain_list


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
        mesh_list.append(Mesh(item, mesh_size, mesh_size, mesh_size))

    return mesh_list

