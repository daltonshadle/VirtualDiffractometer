# **************************************************************************************************
# Name:    equipment_class.py
# Purpose: Module to provides equipment classes for virtual diffraction
# Input:   none
# Output:  Module equipment class definitions for virtual diffraction
# Notes:   none
# **************************************************************************************************


# ********************************************* Imports ********************************************
# Standard Library
import numpy as np
import scipy.constants as sciconst
# My Library


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
    def __init__(self, energy_in, incoming_in):
        self.beamEnergy = energy_in
        self.k_in_lab = ((2 * sciconst.pi) / self.kev_2_angstroms()) * (-1 * incoming_in)

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
    distance = None          # Distance from detector to sample (mm)
    width = None             # Width of detector (pixel)
    height = None            # Height of detector (pixel)
    pixel_density_mm = None  # The amount of pixels per mm (pixel/mm)

    # Constructor
    def __init__(self, dist_in, width_in, height_in, pixel_density_mm_in):
        self.distance = dist_in
        self.width = width_in
        self.height = height_in
        self.pixel_density_mm = pixel_density_mm_in

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



