#!/usr/bin/env python

"""
Reads in preprocessed data from npz files
Takes in total expected events (outside 10-75 MeV range)
Filters data to 10-75 MeV range 
Uses all statistics to make Asimov PDF
Creates toy datasets with Poisson + flux errors
Makes hist of Asimov PDF with a few toy datasets overlayed
Performs a 1D (energy) or 2d (energy + direction) nll fit with minuit
Uses average Minuit errors to calculate statistical precision
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from iminuit import cost, Minuit
from scipy.interpolate import RegularGridInterpolator, interp1d
import os
hep.style.use("ROOT")

##################################
###### configuration 
##################################

PREPROCESSED_DIR = "/nfs/disk1/users/bharris/eos/sim/preprocessed_data"
FIT_SCENARIO = "oxygen"  # oxgyen/gallium
FIT_DIMENSION = "2D"  # 1D/2D

CONFIGS = [
    ('water', '0ft', 10),
    ('water', '0ft', 100),
    ('water', '1ft', 10),
    ('water', '1ft', 100),
    ('water', '3ft', 10),
    ('water', '3ft', 100),
    #('1wbls', '0ft', 10),
    #('1wbls', '0ft', 100),
    #('1wbls', '1ft', 10),
    #('1wbls', '1ft', 100),
    #('1wbls', '3ft', 10),
    #('1wbls', '3ft', 100),
]

N_TOYS = 10
EXPOSURE_TIMES = [0.5, 1.0, 2.0, 3.0]  # Years

# Flux uncertainties (only applied to neutrino channels)
FLUX_ERR = {
    'eES': 0.03,
    'nueO16': 0.03,
    'nueGa71': 0.03,
}

# Energy and direction binning
ENERGY_BINS = np.arange(0, 75+1.25, 1.25)
DIRECTION_BINS = np.linspace(-1, 1, 20)

# Set binning based on dimension
if FIT_DIMENSION == "1D":
    BINNING = ENERGY_BINS
else:
    BINNING = [ENERGY_BINS, DIRECTION_BINS]

# Expected event rates PER YEAR (will be updated with filtered rates)
# No gallium loading
EVENT_RATES_TOTAL = {
    'eES': 219,
    'nueO16': 473,
    'nueGa71': 0, 
    'cosmics': 27000,
    'neutrons': 0 # filled when scaling
}

# For 10% natural gallium loading
#EVENT_RATES_TOTAL = {
#    'eES': 233,
#    'nueO16': 465,
#    'nueGa71': 258,
#    'cosmics': 27000,
#    'neutrons': 0 # will be filled when scaling
#}

EVENT_RATES_FILTERED = {}  # populated after energy range filters

CHANNEL_LABELS = {
    'eES': r"$e$ ES",
    'nueO16': r"$\nu_e$ O-16",
    'nueGa71': r"$\nu_e$ Ga-71",
    'cosmics': "Cosmics",
    'neutrons': "Neutrons"
}

CHANNEL_COLORS = {
    'eES': plt.cm.tab10.colors[0],
    'nueO16': plt.cm.tab10.colors[1],
    'nueGa71': plt.cm.tab10.colors[2],
    'cosmics': plt.cm.tab10.colors[3],
    'neutrons': plt.cm.tab10.colors[4]
}

####################################################################
###### load in preprocessed data from .npz files
####################################################################

def load_preprocessed_channel(detector_name, channel_name):
    filename = f"{detector_name}_{channel_name}.npz"
    filepath = os.path.join(PREPROCESSED_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    data = np.load(filepath)
    energy = data['energy']
    direction = data['direction']
    ntrig = int(data['ntrig'])
    nsim = int(data['nsim'])
    print(f"{channel_name}: {len(energy):,} events from {filename}")
    return energy, direction, ntrig, nsim

def load_neutrons_and_scale(detector_name, shielding, beam_power):
    """Load and scale neutron data."""
    filename = f"{detector_name}_neutrons_{shielding}.npz"
    filepath = os.path.join(PREPROCESSED_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    data = np.load(filepath)
    energy = data['energy']
    direction = data['direction']
    mcke = data['mcke']
    ntrig = int(data['ntrig'])
    nsim = int(data['nsim'])
    # Apply beam power scaling like Richie did
    neutron_spectrum_file = "/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons/neutron_spectra.dat"
    if not os.path.exists(neutron_spectrum_file):
        print(f"    WARNING: Spectrum file not found, using uniform scaling")
        scale_factor = beam_power / 100.0
        N_new = int(len(energy) * scale_factor)
        sampled_indices = np.random.choice(len(energy), size=N_new, replace=True)
        energy_scaled = energy[sampled_indices]
        direction_scaled = direction[sampled_indices]
    else:
        n = np.genfromtxt(neutron_spectrum_file)
        snsyear = 5000 * 60 * 60
        sim_neutrons = nsim / 16.
        expected_neutrons = beam_power * 5000 * 2.8
        bin_edges = np.linspace(0, 200, 201)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        n_expected = np.interp(bin_centers, n[:, 0], n[:, 1])
        n_expected *= expected_neutrons / np.sum(n_expected)
        n_sim_per_bin = sim_neutrons / len(bin_centers)
        n_scaling = n_expected / n_sim_per_bin
        i_scaling = np.interp(mcke, bin_centers, n_scaling)
        N_new = int(np.sum(i_scaling))
        probabilities = i_scaling / np.sum(i_scaling)
        sampled_indices = np.random.choice(len(energy), size=N_new, p=probabilities, replace=True)
        energy_scaled = energy[sampled_indices]
        direction_scaled = direction[sampled_indices]
    print(f"Loaded and scaled {len(energy_scaled):,} neutron events for ({beam_power} MW)")
    return energy_scaled, direction_scaled, ntrig, nsim

def filter_data_to_analysis_range(energy_direction_data):
    """
    Filter data to within MeV range (10 MeV minimum cut?)
    Then scale event rates accordingly
    """
    filtered_data = {}
    filtered_rates = {}
    print(f"Filtering data to analysis range {np.min(ENERGY_BINS), np.max(ENERGY_BINS)}:")
    for key in energy_direction_data:
        energy, direction, ntrig, nsim = energy_direction_data[key]
        # Apply filter
        mask = (energy >= 0) & (energy < 75) & (direction >= -1) & (direction <= 1)
        energy_filtered = energy[mask]
        direction_filtered = direction[mask]
        # Calculate filtered rate
        total_events = len(energy)
        filtered_events = len(energy_filtered)
        fraction = filtered_events / total_events if total_events > 0 else 0
        if key in EVENT_RATES_TOTAL and EVENT_RATES_TOTAL[key] > 0:
            filtered_rate = EVENT_RATES_TOTAL[key] * fraction
        else:
            # For neutrons, calculate from filtered data
            filtered_rate = filtered_events
        filtered_data[key] = (energy_filtered, direction_filtered)
        filtered_rates[key] = filtered_rate
        print(f"{key}: Total Events: {total_events:,} - Filtered Events: {filtered_events:,} events ({100*fraction:.1f}%), rate = {filtered_rate:.1f}/year")
    return filtered_data, filtered_rates

################################
# analysis
################################

def make_toy_datasets_with_poisson_and_flux(filtered_data, filtered_rates, years, ngroups):
    """
    Sample with replacement from all events.
    Apply Poisson fluctuations and flux uncertainties to the number of samples.
    """
    toy_datasets = {key: [] for key in filtered_data.keys()}
    for key in filtered_data.keys():
        energies, directions = filtered_data[key]
        base_rate = filtered_rates[key]
        for i in range(ngroups):
            # Apply flux uncertainty for eES and nueO16
            if key in FLUX_ERR:
                flux_variation = np.random.normal(1.0, FLUX_ERR[key])
                expected_rate = base_rate * flux_variation
            else:
                expected_rate = base_rate
            # Apply Poisson fluctuation
            n_events = np.random.poisson(expected_rate * years)
            # Sample with replacement
            if len(energies) > 0 and n_events > 0:
                idx = np.random.choice(len(energies), size=n_events, replace=True)
                sampled_energies = energies[idx]
                sampled_directions = directions[idx]
            else:
                sampled_energies = np.array([])
                sampled_directions = np.array([])
            toy_datasets[key].append([sampled_energies, sampled_directions])
    return toy_datasets

def normalize_and_scale_asimov(asimov_hist, years, filtered_rates):
    asimov_normalized = {}
    asimov_scaled = {}
    for ch_name, hist in asimov_hist.items():
        # Normalize to PDF
        norm = np.sum(hist)
        if norm > 0:
            asimov_normalized[ch_name] = hist / norm
        else:
            asimov_normalized[ch_name] = hist
        # Scale by expected rate
        asimov_scaled[ch_name] = asimov_normalized[ch_name] * filtered_rates[ch_name] * years
    return asimov_normalized, asimov_scaled

def make_normalized_interpolated_pdf(asimov_normalized):
    if FIT_DIMENSION == "1D":
        # 1D case: just use energy
        bin_centers = 0.5 * (ENERGY_BINS[1:] + ENERGY_BINS[:-1])
        norm_pdf_luts = {}
        pdfs = {}
        for ch_name, hist in asimov_normalized.items():
            # Normalize by bin width
            h = hist.copy()
            dx = np.diff(ENERGY_BINS)
            h = h / dx
            # Create 1D interpolator
            interpolator = interp1d(bin_centers, h, bounds_error=False, fill_value=0, kind='linear')
            norm_pdf_luts[ch_name] = interpolator
            pdfs[ch_name] = h
        return norm_pdf_luts, pdfs, bin_centers
    else:
        # 2D case: energy and direction
        bin_centers = [0.5 * (ENERGY_BINS[1:] + ENERGY_BINS[:-1]), 0.5 * (DIRECTION_BINS[1:] + DIRECTION_BINS[:-1])]
        norm_pdf_luts = {}
        pdfs = {}
        for ch_name, hist in asimov_normalized.items():
            # Normalize each bin
            h = hist.copy()
            # Compute bin widths
            dx = np.diff(ENERGY_BINS)
            dy = np.diff(DIRECTION_BINS)
            # Normalize by bin area
            for i in range(len(dx)):
                for j in range(len(dy)):
                    h[i, j] = h[i, j] / (dx[i] * dy[j])
            # Create interpolator
            interpolator = RegularGridInterpolator(
                (bin_centers[0], bin_centers[1]), h, 
                bounds_error=False, fill_value=0
            )
            norm_pdf_luts[ch_name] = interpolator
            pdfs[ch_name] = h
        return norm_pdf_luts, pdfs, bin_centers

###############################
# plotting
###########################

def plot_asimov_projections(asimov_hist, years, output_path):
    if FIT_DIMENSION == "1D":
        # For 1D, just plot energy histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        en_hists = {ch_name: (hist, ENERGY_BINS) for ch_name, hist in asimov_hist.items()}
        hep.histplot(list(en_hists.values()), stack=False, histtype='step', label=en_hists.keys(), ax=ax)
        ax.semilogy()
        ax.set_xlim(0, 75)
        ax.set_ylabel(f"Events / SNS-Year / {ENERGY_BINS[1]-ENERGY_BINS[0]:.2f} MeV", fontsize=20)
        ax.set_xlabel(f"Reconstructed Electron Energy [MeV]", fontsize=20)
        ax.legend(loc='upper right', ncol=2, fontsize=20)
        ax.set_title("1D Fit (Energy Only)", fontsize=16)
    else:
        # For 2D, plot both projections
        en_hists = {ch_name: (np.sum(hist, axis=1), ENERGY_BINS) for ch_name, hist in asimov_hist.items()}
        dir_hists = {ch_name: (np.sum(hist, axis=0), DIRECTION_BINS) for ch_name, hist in asimov_hist.items()}
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        # Energy projection
        hep.histplot(list(en_hists.values()), stack=False, histtype='step', label=en_hists.keys(), ax=ax[0])
        ax[0].semilogy()
        ax[0].set_xlim(0, 75)
        ax[0].set_ylabel(f"Events / SNS-Year / {ENERGY_BINS[1]-ENERGY_BINS[0]:.2f} MeV", fontsize=20)
        ax[0].set_xlabel(f"Reconstructed Electron Energy [MeV]", fontsize=20)
        ax[0].legend(loc='upper right', ncol=2, fontsize=20)
        # Direction projection
        hep.histplot(list(dir_hists.values()), stack=False, histtype='step', label=dir_hists.keys(), ax=ax[1])
        ax[1].set_xlim(-1, 1)
        ax[1].set_ylabel(f"Events / SNS-Year / {DIRECTION_BINS[1]-DIRECTION_BINS[0]:.2f}", fontsize=20)
        ax[1].set_xlabel(f"Reconstructed Electron Direction [cos$\\theta$]", fontsize=20)
        ax[1].semilogy()
        ax[1].legend(loc='upper right', ncol=2, fontsize=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved Asimov hists: {output_path}")
    plt.close()

def plot_asimov_and_fit_group_projections(asimov_hist, fitgroups_hist, years, output_path, n_toys_to_plot=10):
    """
    Plot Asimov and first N toy datasets overlaid (or else it gets too messy
    """
    if FIT_DIMENSION == "1D":
        # 1D case
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # Plot Asimov (solid lines)
        for key in asimov_hist.keys():
            hep.histplot(asimov_hist[key], bins=ENERGY_BINS, histtype='step', label=f"Asimov {key}", color=CHANNEL_COLORS.get(key, 'black'), linewidth=2, ax=ax)
        # Plot only first N toys (dashed lines)
        n_to_plot = min(n_toys_to_plot, len(fitgroups_hist))
        for group_idx in range(n_to_plot):
            fit_hist = fitgroups_hist[group_idx]
            for key in fit_hist.keys():
                # Only add label for first toy
                label = f"Toy {key}" if group_idx == 0 else None
                hep.histplot(fit_hist[key], bins=ENERGY_BINS, histtype='step', linestyle='--', color=CHANNEL_COLORS.get(key, 'black'), alpha=0.3, label=label, ax=ax)
        ax.set_xlabel("Reconstructed Energy [MeV]", fontsize=20)
        ax.set_ylabel("Events", fontsize=20)
        ax.semilogy()
        ax.set_xlim(0, 75)
        ax.legend(fontsize=12)
        ax.set_title(f"First {n_to_plot} Toys (1D Fit)", fontsize=14)
    else:
        # 2D case
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        # Plot Asimov (solid lines)
        for key in asimov_hist.keys():
            asimov_energy = np.sum(asimov_hist[key], axis=1)
            asimov_direction = np.sum(asimov_hist[key], axis=0)
            hep.histplot(asimov_energy, bins=ENERGY_BINS, histtype='step', label=f"Asimov {key}", color=CHANNEL_COLORS.get(key, 'black'), linewidth=2, ax=ax[0])
            hep.histplot(asimov_direction, bins=DIRECTION_BINS, histtype='step', label=f"Asimov {key}", color=CHANNEL_COLORS.get(key, 'black'), linewidth=2, ax=ax[1])
        # Plot only first N toys (dashed lines)
        n_to_plot = min(n_toys_to_plot, len(fitgroups_hist))
        for group_idx in range(n_to_plot):
            fit_hist = fitgroups_hist[group_idx]
            for key in fit_hist.keys():
                fit_energy = np.sum(fit_hist[key], axis=1)
                fit_direction = np.sum(fit_hist[key], axis=0)
                # Only add label for first toy
                label = f"Toy {key}" if group_idx == 0 else None
                hep.histplot(fit_energy, bins=ENERGY_BINS, histtype='step', linestyle='--', color=CHANNEL_COLORS.get(key, 'black'), alpha=0.3, label=label, ax=ax[0])
                hep.histplot(fit_direction, bins=DIRECTION_BINS, histtype='step', linestyle='--', color=CHANNEL_COLORS.get(key, 'black'), alpha=0.3, label=label, ax=ax[1])
        ax[0].set_xlabel("Reconstructed Energy [MeV]", fontsize=20)
        ax[0].set_ylabel("Events", fontsize=20)
        ax[0].semilogy()
        ax[0].set_xlim(0, 75)
        ax[0].legend(fontsize=12)
        ax[0].set_title(f"First {n_to_plot} Toys", fontsize=14)
        ax[1].set_xlabel("Reconstructed Direction [cos(theta)]", fontsize=20)
        ax[1].set_ylabel("Events", fontsize=20)
        ax[1].semilogy()
        ax[1].set_xlim(-1, 1)
        ax[1].legend(fontsize=12)
        ax[1].set_title(f"First {n_to_plot} Toys", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved first {n_to_plot} toy hists to: {output_path}")
    plt.close()

############################
# fitting
############################

def fit_with_extended_binned_nll(fit_data, channels_to_fit, norm_pdf_luts, years, total_events):
    '''
    This works for 1d/2d fit for oxygen/gallium... so it is messy 
    Adds a penalty term for the total number of fitted events
    '''
    if FIT_DIMENSION == "1D":

        if FIT_SCENARIO == "oxygen":

            # combine everything
            def pdf_cut(xe, nES, nNeutrons, nCosmics, nO16):
                eES_vals = norm_pdf_luts['eES'](xe)
                neutrons_vals = norm_pdf_luts['neutrons'](xe)
                cosmics_vals = norm_pdf_luts['cosmics'](xe)
                nueO16_vals = norm_pdf_luts['nueO16'](xe)
                result = (
                    nES * eES_vals +
                    nNeutrons * neutrons_vals + 
                    nCosmics * cosmics_vals + 
                    nO16 * nueO16_vals)
                return result
            # nll with penalty for total number of events
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16)
                total_fit = nES + nNeutrons + nCosmics + nO16
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            # set fit guesses and limits 
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16'] = (0, None)
            
        elif FIT_SCENARIO == "gallium":

            # combine everything
            def pdf_cut(xe, nES, nNeutrons, nCosmics, nO16, nGa71):
                eES_vals = norm_pdf_luts['eES'](xe)
                neutrons_vals = norm_pdf_luts['neutrons'](xe)
                cosmics_vals = norm_pdf_luts['cosmics'](xe)
                nueO16_vals = norm_pdf_luts['nueO16'](xe)
                nueGa71_vals = norm_pdf_luts['nueGa71'](xe)
                result = (nES * eES_vals +
                         nNeutrons * neutrons_vals +
                         nCosmics * cosmics_vals +
                         nO16 * nueO16_vals +
                         nGa71 * nueGa71_vals)
                return result
            # nll with penalty for total number of events
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16, nGa71):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16, nGa71)
                total_fit = nES + nNeutrons + nCosmics + nO16 + nGa71
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            # set fit guesses and limits 
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years,
                      nGa71=EVENT_RATES_FILTERED['nueGa71']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71'] = (0, None)
    
    else:

        if FIT_SCENARIO == "oxygen":

            # combine everything
            def pdf_cut(xe_ye, nES, nNeutrons, nCosmics, nO16):
                xe, ye = xe_ye
                original_shape = xe.shape
                points = np.column_stack([xe.ravel(), ye.ravel()])
                eES_vals = norm_pdf_luts['eES'](points).reshape(original_shape)
                neutrons_vals = norm_pdf_luts['neutrons'](points).reshape(original_shape)
                cosmics_vals = norm_pdf_luts['cosmics'](points).reshape(original_shape)
                nueO16_vals = norm_pdf_luts['nueO16'](points).reshape(original_shape)
                result = (nES * eES_vals +
                         nNeutrons * neutrons_vals +
                         nCosmics * cosmics_vals +
                         nO16 * nueO16_vals)
                return result
            # nll with penalty term for total number of events
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16)
                total_fit = nES + nNeutrons + nCosmics + nO16
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            # set fit guesses and limits 
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16'] = (0, None)
            
        elif FIT_SCENARIO == "gallium":

            # combine everything
            def pdf_cut(xe_ye, nES, nNeutrons, nCosmics, nO16, nGa71):
                xe, ye = xe_ye
                original_shape = xe.shape
                points = np.column_stack([xe.ravel(), ye.ravel()])
                eES_vals = norm_pdf_luts['eES'](points).reshape(original_shape)
                neutrons_vals = norm_pdf_luts['neutrons'](points).reshape(original_shape)
                cosmics_vals = norm_pdf_luts['cosmics'](points).reshape(original_shape)
                nueO16_vals = norm_pdf_luts['nueO16'](points).reshape(original_shape)
                nueGa71_vals = norm_pdf_luts['nueGa71'](points).reshape(original_shape)
                result = (nES * eES_vals +
                         nNeutrons * neutrons_vals +
                         nCosmics * cosmics_vals +
                         nO16 * nueO16_vals +
                         nGa71 * nueGa71_vals)
                return result
            # nll with penalty term for total number of events
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16, nGa71):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16, nGa71)
                total_fit = nES + nNeutrons + nCosmics + nO16 + nGa71
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            # set fit guesses and limits 
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years,
                      nGa71=EVENT_RATES_FILTERED['nueGa71']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71'] = (0, None)

    # do the fit 
    m.migrad()
    m.hesse()
    # in case it fails
    try:
        m.minos()
    except:
        print("    MINOS failed, using HESSE errors only")
    return m

def run_scenario_analysis(channel_cache, shielding, beam_power, detector_name):
    '''
    do this for each fit scenario
    '''
    print("####################################")
    print(f"SCENARIO: {detector_name}_{beam_power}MW_{shielding}")
    print("####################################")
    # Load neutrons
    energy_direction_data = {}
    for ch in channel_cache:
        energy, direction, ntrig, nsim = channel_cache[ch]
        energy_direction_data[ch] = (energy, direction, ntrig, nsim)
    try:
        neutron_data = load_neutrons_and_scale(detector_name, shielding, beam_power)
        energy_direction_data['neutrons'] = neutron_data
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return None
    # Filter data to MeV analysis range 
    filtered_data, filtered_rates = filter_data_to_analysis_range(energy_direction_data)
    # Update global filtered rates
    EVENT_RATES_FILTERED.update(filtered_rates)
    # Create Asimov histograms from ALL filtered data
    print(f"\nCreating Asimov PDF ({FIT_DIMENSION}) from all filtered data:")
    asimov_data = filtered_data
    if FIT_DIMENSION == "1D":
        asimov_hist = {key: np.histogram(asimov_data[key][0], BINNING)[0] for key in filtered_data}
    else:
        asimov_hist = {key: np.histogram2d(asimov_data[key][0], asimov_data[key][1], BINNING)[0] for key in filtered_data}
    for key in asimov_hist:
        print(f"{key}: {np.sum(asimov_hist[key]):.0f} events in histogram")
    # Determine channels to fit
    if FIT_SCENARIO == "oxygen":
        channels = ['nES', 'nNeutrons', 'nCosmics', 'nO16']
        signal_channel = 'nO16'
    elif FIT_SCENARIO == "gallium":
        channels = ['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71']
        signal_channel = 'nGa71'
    # Results storage
    all_results = {years: [] for years in EXPOSURE_TIMES}
    for years in EXPOSURE_TIMES:
        print(f"\nEXPOSURE: {years} years")
        print("####################################")
        # Make toy datasets with Poisson + flux errors
        toy_datasets = make_toy_datasets_with_poisson_and_flux(filtered_data, filtered_rates, years, N_TOYS)
        
        # Make toy histograms
        toygroups_hist = []
        for i in range(N_TOYS):
            if FIT_DIMENSION == "1D":
                toy_hist = {key: np.histogram(toy_datasets[key][i][0], BINNING)[0] 
                           for key in filtered_data}
            else:
                toy_hist = {key: np.histogram2d(toy_datasets[key][i][0], toy_datasets[key][i][1], BINNING)[0] 
                           for key in filtered_data}
            toygroups_hist.append(toy_hist)
        
        # Normalize and scale Asimov
        asimov_normalized, asimov_scaled = normalize_and_scale_asimov(asimov_hist, years, filtered_rates)
        
        # Make interpolated PDFs
        norm_pdf_luts, pdfs, bin_centers = make_normalized_interpolated_pdf(asimov_normalized)
        
        # Plot Asimov projections (once per scenario)
        if years == EXPOSURE_TIMES[0]:
            output_path = f'/nfs/disk1/users/bharris/eos/sim/eos-sns-analysis/oxygen_analysis/new/asimov_projections_{detector_name}_{beam_power}MW_{shielding}_{FIT_DIMENSION}.png'
            plot_asimov_projections(asimov_scaled, years, output_path)
        
        # Plot Asimov + first 10 toys overlaid
        output_path = f'/nfs/disk1/users/bharris/eos/sim/eos-sns-analysis/oxygen_analysis/new/asimov_toys_{detector_name}_{beam_power}MW_{shielding}_{years}yr_{FIT_DIMENSION}.png'
        plot_asimov_and_fit_group_projections(asimov_scaled, toygroups_hist, years, output_path, n_toys_to_plot=10)
        
        # Fit each toy dataset
        total_events = sum(filtered_rates[ch] for ch in filtered_rates.keys())
        
        fit_results = []
        for i in range(N_TOYS):
            # Sum all channels to make total data histogram
            fit_data = np.sum([toygroups_hist[i][ch] for ch in filtered_data.keys()], axis=0)
            
            if i == 0:
                
                print(f"\nToy dataset 0:")
                for ch in filtered_data.keys():
                    print(f"{ch}: {np.sum(toygroups_hist[i][ch]):.0f}")
            
            try:
                m = fit_with_extended_binned_nll(fit_data, channels, norm_pdf_luts, years, total_events)
                fit_results.append(m)
                
                if i == 0:
                    print(f"\nToy dataset 0 fit result:")
                    for ch in channels:
                        print(f"{ch}: {m.values[ch]:.1f} ± {m.errors[ch]:.1f}")
            except Exception as e:
                    print(f"  ERROR: Fit {i+1} failed: {e}")
                    continue
        # Store results 
        if len(fit_results) == 0:
            print(f"\n  ERROR: All fits failed for {years} year exposure!")
            continue
        ch_idx = channels.index(signal_channel)
        true_val = filtered_rates[channel_mapping[signal_channel]]*years
        for i, m in enumerate(fit_results):
            all_results[years].append({
                'fitted': m.values[signal_channel],  # Use raw fitted value
                'error': m.errors[signal_channel],    # Use raw Minuit error
                'valid': m.valid,
                'true_value': true_val
            })
        # Print summary
        signal_vals = [m.values[signal_channel] for m in fit_results if m.valid]
        signal_err  = [m.errors[signal_channel] for m in fit_results if m.valid]
        if len(signal_vals) > 0:
            mean_val = np.mean(signal_vals)
            mean_err = np.mean(signal_err)
            true_val = filtered_rates[channel_mapping[signal_channel]] * years
            bias = mean_val - true_val

            # Bias-corrected RMS of fitted values around true value
            corrected_rms = np.sqrt(np.mean([(v - bias - true_val)**2 for v in signal_vals]))

            print(f"\nSummary of {N_TOYS} {signal_channel} toy fit results:")
            print(f"Mean fitted val : {mean_val:.1f}")
            print(f"Mean fitted err : {mean_err:.1f}")
            print(f"Mean fitted err / truth : ({100*mean_err/true_val:.2f}%) <-- minuit statistical error")
            print(f"RMS fitted val  : {np.std(signal_vals):.1f}")
            print(f"True value      : {true_val:.1f}")
            print(f"Bias (mean - true): {bias:.1f}")
            print(f"Bias-corrected RMS around truth: {corrected_rms:.1f} ({100*corrected_rms/true_val:.2f}%)  <-- bias corrected error")
 
    return all_results, signal_channel


# Map minuit parameter names to channel names
channel_mapping = {
    'nES': 'eES',
    'nNeutrons': 'neutrons',
    'nCosmics': 'cosmics',
    'nO16': 'nueO16',
    'nGa71': 'nueGa71'
}

##############################
# main 
##############################

if __name__ == "__main__":

    print(f"Fit scenario: {FIT_SCENARIO}")
    print(f"Fit dimension: {FIT_DIMENSION}")
    print(f"N_TOYS: {N_TOYS}")
    print(f"Exposure times: {EXPOSURE_TIMES}")
    all_results_by_config = {}
    for detector_name, shielding, beam_power in CONFIGS:
        # Load channels
        channel_cache = {}
        channels_to_load = ['nueGa71', 'eES', 'cosmics', 'nueO16'] if FIT_SCENARIO == "gallium" else ['eES', 'cosmics', 'nueO16']
        for channel_name in channels_to_load:
            try:
                data = load_preprocessed_channel(detector_name, channel_name)
                channel_cache[channel_name] = data
            except FileNotFoundError as e:
                print(f"    ERROR: {e}")
                continue
        # Run analysis
        results = run_scenario_analysis(channel_cache, shielding, beam_power, detector_name)
        if results is not None:
            results_dict, signal_channel = results
            all_results_by_config[f"{detector_name}_{shielding}_{beam_power}MW"] = results_dict

    print("Plotting precision curves...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Print table header
    print("\n" + "=" * 120)
    print(f"{'Config':<30} {'Years':<8} {'Minuit Stat (%)':>18} {'Avg Bias':>15} {'Bias-Corr RMS (%)':>20}")
    print("=" * 120)
    
    for config_name, results in all_results_by_config.items():
        precisions = []
        
        for years in EXPOSURE_TIMES:
            if years in results and len(results[years]) > 0:
                # Get errors and fitted values from valid fits
                errors = [r['error'] for r in results[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in results[years] if r['valid']]
                
                if len(errors) > 0:
                    # Calculate metrics
                    avg_error = np.mean(errors)
                    avg_fitted = np.mean(fitted_vals)
                    
                    # Get true value for this configuration
                    true_val = results[years][0]['true_value']
                    
                    # Calculate bias
                    bias = avg_fitted - true_val
                    
                    # Calculate bias-corrected RMS around truth
                    corrected_rms = np.sqrt(np.mean([(v - bias - true_val)**2 for v in fitted_vals]))
                    
                    # Minuit statistical precision
                    minuit_stat_precision = 100 * avg_error / avg_fitted
                    
                    # Bias-corrected RMS as percentage of true value
                    corrected_rms_percent = 100 * corrected_rms / true_val
                    
                    # Append for plotting
                    precisions.append(corrected_rms_percent)
                    
                    # Print table row
                    print(f"{config_name:<30} {years:<8.1f} {minuit_stat_precision:>18.2f} {bias:>15.1f} {corrected_rms_percent:>20.2f}")
                else:
                    precisions.append(np.nan)
            else:
                precisions.append(np.nan)
        
        ax.plot(EXPOSURE_TIMES, precisions, 'o-', label=config_name, linewidth=2, markersize=8)
    
    print("=" * 120)
    
    ax.set_xlabel("Years of Exposure", fontsize=14)
    ax.set_ylabel(f"Bias-Corrected RMS on {signal_channel} (%)", fontsize=14)
    ax.set_title(f"Statistical Precision vs. Exposure Time\n{FIT_SCENARIO.capitalize()} Sensitivity ({FIT_DIMENSION} Fit - Bias-Corrected RMS)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    output_path = f'/nfs/disk1/users/bharris/eos/sim/eos-sns-analysis/oxygen_analysis/new/precision_curves_{FIT_SCENARIO}_{FIT_DIMENSION}.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved precision curves: {output_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
