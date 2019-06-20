# **************************************************************************************************
# Name:    io_functions.py
# Purpose: Defines input and output functions for this project
# Input:   none
# Output:  input and output functions
# Notes:   none
# **************************************************************************************************

# ********************************************* Imports ********************************************
# Standard Library
import numpy as np
import pathlib
import csv
# My Library
import definitions
import classes.sample_class as sample_class
import utils.math_functions as math_func

# *************************************** Variable Definitions *************************************
DATA_PATH = pathlib.Path(definitions.ROOT_DIR + "/data/")
IMG_PATH = pathlib.Path(definitions.ROOT_DIR + "/img/")
SRC_PATH = pathlib.Path(definitions.ROOT_DIR + "/src/")
TEST_PATH = pathlib.Path(definitions.ROOT_DIR + "/test/")


# *************************************** Function Definitions *************************************
def read_hkl_from_csv(filename):
    # **********************************************************************************************
    # Name:    read_hkl_from_csv
    # Purpose: function that reads the hkl vector list from csv and returns as list
    # Input:   filename (string) - filename of the data to load, csv must be in n x 3 format
    # Output:  hkl_list (n x 3 matrix) - matrix that holds all the hkl vectors from csv file
    # Notes:   none
    # **********************************************************************************************

    # initialize return list and path
    hkl_list = np.empty((0, 3))
    file_path = DATA_PATH / filename

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


def read_grains_from_csv(filename):
    # **********************************************************************************************
    # Name:    read_grains_from_csv
    # Purpose: function that reads grain data from csv and returns as list of grains
    # Input:   filename (string) - filename of data to load, csv must in n x 13 format
    # Output:  grain_list (list) - matrix that holds n grains from csv file
    # Notes:   none
    # **********************************************************************************************

    # initialize return list and path
    grain_list = []
    file_path = DATA_PATH / filename

    # open csv storing hkl vectors
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        temp_cell = sample_class.UnitCell(np.array([4, 4, 4, 90, 90, 90]))

        # iterate over all rows in csv and add to grain list
        for row in csv_reader:
            if line_count != 0:
                dim_vec = np.array([float(row[0]), float(row[1]), float(row[2])])
                com_vec = np.array([float(row[3]), float(row[4]), float(row[5])])
                quat_vec = np.array([float(row[6]), float(row[7]), float(row[8]), float(row[9])])
                orient_vec = np.array([float(row[10]), float(row[11]), float(row[1])])

                # normalize vectors
                quat_vec = math_func.normalize(quat_vec)
                orient_vec = math_func.normalize(orient_vec)

                # Grain (unitCell, dimension, COM, orientation)
                temp_grain = sample_class.Grain(temp_cell, dim_vec, com_vec, quat_vec)
                temp_grain.vector2quat(orient_vec)

                # add to list
                grain_list.append(temp_grain)
            line_count += 1
        print(f'Processed {line_count} grains.')
    return grain_list
