#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# data paths (relative to project root)
DATA_ROOT = PROJECT_ROOT.parent.parent / "sim"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed_data_merged"
NEUTRON_SPECTRUM_FILE = DATA_ROOT / "outputs" / "sns" / "neutrons" / "neutron_spectra.dat"

# output paths (within project)
RESULTS_DIR = PROJECT_ROOT / "results"
HISTS_DIR = PROJECT_ROOT / "hists"
LOG_DIR = PROJECT_ROOT / "log"

# ensure output directories exist
for directory in [RESULTS_DIR, HISTS_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# energy reconstruction factors
ALPHA_WATER = 0.04138615778199661
ALPHA_1WBLS = 0.010945997573964417

# analysis energy range
ENERGY_MIN = 0.0
ENERGY_MAX = 75.0

# binning
ENERGY_BINS = np.arange(ENERGY_MIN, ENERGY_MAX + 1.25, 1.25)
#ENERGY_BINS = np.arange(ENERGY_MIN, ENERGY_MAX + 5, 5)
#DIRECTION_BINS = np.linspace(-1, 1, 16)
DIRECTION_BINS = np.linspace(-1, 1, 51)

FIT_SCENARIO = "oxygen" # oxygen/gallium
FIT_DIMENSION = "2D" # 1D/2D

# flux uncertainties (only applied to neutrino channels)
FLUX_ERR = {
    'eES': 0.03,
    'nueO16': 0.03,
    'nueGa71': 0.03,
}

# expected event rates per year (baseline - no gallium)
EVENT_RATES_TOTAL_NO_GA = {
    'eES': 219,
    'nueO16': 473,
    'nueGa71': 0, 
    'cosmics': 27000,
    'neutrons': 0  # filled when scaling
}

# for 10% natural gallium loading
EVENT_RATES_TOTAL_WITH_GA = {
    'eES': 233,
    'nueO16': 465,
    'nueGa71': 258,
    'cosmics': 27000,
    'neutrons': 0
}

EVENT_RATES_TOTAL = EVENT_RATES_TOTAL_NO_GA.copy()

# detector configurations: (detector, shielding, neutrons_per_mw)
CONFIGS = [
    ('water', '0ft', 10),
    ('water', '0ft', 100),
    ('water', '1ft', 10),
    ('water', '1ft', 100),
    ('water', '3ft', 10),
    ('water', '3ft', 100),
    ('1wbls', '0ft', 10),
    ('1wbls', '0ft', 100),
    ('1wbls', '1ft', 10),
    ('1wbls', '1ft', 100),
    ('1wbls', '3ft', 10),
    ('1wbls', '3ft', 100),
]

# number of toy datasets
N_TOYS = 10 # increase for production (e.g. 1000)

# exposure times to analyze (years)
EXPOSURE_TIMES = [0.5, 1.0, 2.0, 3.0]

# fraction of data for asimov pdf (rest for toys)
# 0.5 = 50% for asimov, 50% for toy sampling (no overlap!)
ASIMOV_FRACTION = 0.5

# smoothing configuration for ASIMOV HISTOGRAMS ONLY (applied AFTER splitting)
# This is the statistically correct approach:
# - Split data first into asimov and toy pools
# - Create histograms from raw events in both pools
# - Smooth ONLY the asimov histogram (not the toy pool!)
# - Toy datasets should retain natural statistical fluctuations
SMOOTH_ASIMOV = {
    'enabled': False,  # toggle on/off
    'channels': ['neutrons', 'cosmics'],  # which ASIMOV histograms to smooth
    'method': 'gaussian',  # 'gaussian' (best for 1D & 2D), 'kde', 'spline', 'savgol', 'exponential' (1D only)
    'params': {
        'gaussian': {'sigma': 0.2},  # NEW: Recommended method - works for 1D and 2D!
        'spline': {'smoothness': 1},
        'kde': {'bandwidth': 'scott'},
        'savgol': {'window': 11, 'polyorder': 3},
        'exponential': {}  # No additional parameters for exponential
    }
}

# channel display labels
CHANNEL_LABELS = {
    'eES': r"$e$ ES",
    'nueO16': r"$\nu_e$ O-16",
    'nueGa71': r"$\nu_e$ Ga-71",
    'cosmics': "Cosmics",
    'neutrons': "Neutrons"
}

# channel colors
CHANNEL_COLORS = {
    'eES': plt.cm.tab10.colors[0],
    'nueO16': plt.cm.tab10.colors[1],
    'nueGa71': plt.cm.tab10.colors[2],
    'cosmics': plt.cm.tab10.colors[3],
    'neutrons': plt.cm.tab10.colors[4],
    'extra0': plt.cm.tab10.colors[5],
    'extra1': plt.cm.tab10.colors[6],
    'extra2': plt.cm.tab10.colors[7]
}

# signal channel labels
SIGNAL_LABELS = {
    'nO16': 'CC nue O16',
    'nGa71': 'CC nue Ga71'
}

# map minuit parameter names to channel names
CHANNEL_MAPPING = {
    'nES': 'eES',
    'nNeutrons': 'neutrons',
    'nCosmics': 'cosmics',
    'nO16': 'nueO16',
    'nGa71': 'nueGa71'
}

# sns parameters
SNS_HOURS_PER_YEAR = 5000
SNS_BEAM_MW = 2.8  # beam power in mw

# neutron simulation parameters  
NEUTRON_SIM_AREA_M2 = 16.0  # sim grid is 4x4 m = 16 m^2

def get_binning(dimension):
    if dimension == "1D":
        return ENERGY_BINS
    else:
        return [ENERGY_BINS, DIRECTION_BINS]

def get_channels_for_scenario(scenario):
    if scenario == "oxygen":
        channels = ['nES', 'nNeutrons', 'nCosmics', 'nO16']
        signal_channel = 'nO16'
    elif scenario == "gallium":
        channels = ['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71']
        signal_channel = 'nGa71'
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return channels, signal_channel

def get_channels_to_load(scenario):
    base_channels = ['eES', 'cosmics', 'nueO16']
    if scenario == "gallium":
        return ['nueGa71'] + base_channels
    return base_channels

def validate_paths():
    if not PREPROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"preprocessed data directory not found: {PREPROCESSED_DIR}\n"
            f"please run preprocess_data.py first"
        )
    if not NEUTRON_SPECTRUM_FILE.exists():
        print(f"warning: neutron spectrum file not found: {NEUTRON_SPECTRUM_FILE}")
        print("neutron scaling will use uniform scaling")
