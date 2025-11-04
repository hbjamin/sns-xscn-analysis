#!/usr/bin/env python
"""
Single Config Fitter - Parallelized Version 

Reads in preprocessed data from npz files
Takes in total expected events 
Filters data to 10-75 MeV range 
Uses all statistics to make Asimov PDF
Creates toy datasets with Poisson + flux errors
Performs a 1D (energy) or 2d (energy + direction) nll fit with minuit
Uses bias-corrected RMS to calculate statistical precision
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from iminuit import cost, Minuit
from scipy.interpolate import RegularGridInterpolator, interp1d
import os
import sys
import pickle
import gc  # For explicit garbage collection

hep.style.use("ROOT")

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

if len(sys.argv) != 6:
    print("Usage: python test_single_config.py <detector> <shielding> <beam_power> <fit_scenario> <fit_dimension>")
    print("Example: python test_single_config.py water 0ft 100 oxygen 2D")
    sys.exit(1)

DETECTOR_NAME = sys.argv[1]  # e.g., 'water', '1wbls'
SHIELDING = sys.argv[2]      # e.g., '0ft', '1ft', '3ft'
BEAM_POWER = int(sys.argv[3])  # e.g., 10, 100
FIT_SCENARIO = sys.argv[4]   # 'oxygen' or 'gallium'
FIT_DIMENSION = sys.argv[5]  # '1D' or '2D'

print("="*80)
print("RUNNING SINGLE CONFIG ANALYSIS")
print("="*80)
print(f"Detector: {DETECTOR_NAME}")
print(f"Shielding: {SHIELDING}")
print(f"Beam Power: {BEAM_POWER} MW")
print(f"Fit Scenario: {FIT_SCENARIO}")
print(f"Fit Dimension: {FIT_DIMENSION}")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

PREPROCESSED_DIR = "/nfs/disk1/users/bharris/eos/sim/preprocessed_data"
RESULTS_DIR = "/nfs/disk1/users/bharris/eos/analysis/sns-xscn-analysis/results"
PLOTS_DIR = "/nfs/disk1/users/bharris/eos/analysis/sns-xscn-analysis"
neutron_spectrum_file = "/nfs/disk1/users/bharris/eos/sim/outputs/sns/neutrons/neutron_spectra.dat"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TOYS = 1000 
EXPOSURE_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Years

# Flux uncertainties (only applied to neutrino channels)
FLUX_ERR = {
    'eES': 0.03,
    'nueO16': 0.03,
    'nueGa71': 0.03,
}

# Energy and direction binning
ENERGY_BINS = np.arange(0, 75, 1.25)
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
    'neutrons': 0 # will be filled when scaling
}

# For 10% natural gallium loading - uncomment if needed
#EVENT_RATES_TOTAL = {
#    'eES': 233,
#    'nueO16': 465,
#    'nueGa71': 258,
#    'cosmics': 27000,
#    'neutrons': 0 # will be filled when scaling
#}

EVENT_RATES_FILTERED = {}  # Will be populated after filtering

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

# ============================================================================
# FAST DATA LOADING
# ============================================================================

def load_preprocessed_channel(detector_name, channel_name):
    """Load preprocessed channel data from .npz file."""
    filename = f"{detector_name}_{channel_name}.npz"
    filepath = os.path.join(PREPROCESSED_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = np.load(filepath)
    energy = data['energy']
    direction = data['direction']
    ntrig = int(data['ntrig'])
    nsim = int(data['nsim'])
    
    print(f"    Loaded {len(energy):,} events from {filename}")
    
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
    
    # Apply beam power scaling
    
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
        
        N_new = int(np.sum(i_scaling) * 2)
        probabilities = i_scaling / np.sum(i_scaling)
        sampled_indices = np.random.choice(len(energy), size=N_new, p=probabilities, replace=True)
        
        energy_scaled = energy[sampled_indices]
        direction_scaled = direction[sampled_indices]
    
    print(f"    Loaded and scaled {len(energy_scaled):,} neutron events ({beam_power} MW)")
    
    return energy_scaled, direction_scaled, ntrig, nsim


def filter_data_to_analysis_range(energy_direction_data):
    """
    Filter all data to analysis range: 0 < energy < 75 MeV, -1 <= direction <= 1.
    Returns filtered data and calculates filtered event rates.
    """
    filtered_data = {}
    filtered_rates = {}
    
    print(f"\nFiltering data to analysis range (0-75 MeV, -1 to 1 cos(theta)):")
    
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
            filtered_rate = filtered_events / 2  # Divide by 2 like original script
        
        filtered_data[key] = (energy_filtered, direction_filtered)
        filtered_rates[key] = filtered_rate
        
        print(f"  {key}: {total_events:,} → {filtered_events:,} events ({100*fraction:.1f}%), rate = {filtered_rate:.1f}/year")
    
    return filtered_data, filtered_rates


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

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
    """Normalize Asimov histograms (works for both 1D and 2D)."""
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
    """Create interpolated PDFs for fitting (handles both 1D and 2D)."""
    
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
            interpolator = interp1d(
                bin_centers, h,
                bounds_error=False, fill_value=0,
                kind='linear'
            )
            
            norm_pdf_luts[ch_name] = interpolator
            pdfs[ch_name] = h
        
        return norm_pdf_luts, pdfs, bin_centers
        
    else:
        # 2D case: energy and direction
        bin_centers = [
            0.5 * (ENERGY_BINS[1:] + ENERGY_BINS[:-1]),
            0.5 * (DIRECTION_BINS[1:] + DIRECTION_BINS[:-1])
        ]
        
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


# ============================================================================
# FITTING
# ============================================================================

def fit_with_extended_binned_nll(fit_data, channels_to_fit, norm_pdf_luts, years, total_events, debug=False):
    """Fit using ExtendedBinnedNLL (handles both 1D and 2D)."""
    if debug:
        print(f"      fit_data shape: {fit_data.shape}, sum: {np.sum(fit_data):.0f}")
        print(f"      channels_to_fit: {channels_to_fit}")
        print(f"      years: {years}, total_events: {total_events}")
    
    # Define PDF function based on dimension
    if FIT_DIMENSION == "1D":
        # 1D case: energy only
        if FIT_SCENARIO == "oxygen":
            def pdf_cut(xe, nES, nNeutrons, nCosmics, nO16):
                eES_vals = norm_pdf_luts['eES'](xe)
                neutrons_vals = norm_pdf_luts['neutrons'](xe)
                cosmics_vals = norm_pdf_luts['cosmics'](xe)
                nueO16_vals = norm_pdf_luts['nueO16'](xe)
                
                result = (nES * eES_vals +
                         nNeutrons * neutrons_vals +
                         nCosmics * cosmics_vals +
                         nO16 * nueO16_vals)
                
                return result
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(
                    nES, nNeutrons, nCosmics, nO16
                )
                total_fit = nES + nNeutrons + nCosmics + nO16
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16'] = (0, None)
            
        elif FIT_SCENARIO == "gallium":
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
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16, nGa71):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(
                    nES, nNeutrons, nCosmics, nO16, nGa71
                )
                total_fit = nES + nNeutrons + nCosmics + nO16 + nGa71
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years,
                      nGa71=EVENT_RATES_FILTERED['nueGa71']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71'] = (0, None)
    
    else:
        # 2D case: energy and direction
        if FIT_SCENARIO == "oxygen":
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
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(
                    nES, nNeutrons, nCosmics, nO16
                )
                total_fit = nES + nNeutrons + nCosmics + nO16
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16'] = (0, None)
            
        elif FIT_SCENARIO == "gallium":
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
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16, nGa71):
                nll = cost.ExtendedBinnedNLL(fit_data, BINNING, pdf_cut, use_pdf="approximate")(
                    nES, nNeutrons, nCosmics, nO16, nGa71
                )
                total_fit = nES + nNeutrons + nCosmics + nO16 + nGa71
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=EVENT_RATES_FILTERED['eES']*years,
                      nNeutrons=EVENT_RATES_FILTERED['neutrons']*years,
                      nCosmics=EVENT_RATES_FILTERED['cosmics']*years,
                      nO16=EVENT_RATES_FILTERED['nueO16']*years,
                      nGa71=EVENT_RATES_FILTERED['nueGa71']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71'] = (0, None)
    
    # Fit
    m.migrad()
    m.hesse()
    try:
        m.minos()
    except:
        if debug:
            print("    MINOS failed, using HESSE errors only")
    
    return m


# Map minuit parameter names to channel names
channel_mapping = {
    'nES': 'eES',
    'nNeutrons': 'neutrons',
    'nCosmics': 'cosmics',
    'nO16': 'nueO16',
    'nGa71': 'nueGa71'
}


def run_scenario_analysis(channel_cache, shielding, beam_power, detector_name):
    """Run full analysis for one scenario."""
    
    print(f"\n{'='*80}")
    print(f"SCENARIO: {detector_name}_{beam_power}MW_{shielding}")
    print(f"{'='*80}")
    
    # Load neutrons
    print(f"\nLoading neutrons:")
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
    
    # Filter data to analysis range (0-75 MeV)
    filtered_data, filtered_rates = filter_data_to_analysis_range(energy_direction_data)
    
    # Update global filtered rates
    EVENT_RATES_FILTERED.update(filtered_rates)
    
    # Create Asimov histograms from ALL filtered data
    print(f"\nCreating Asimov PDF from all filtered data ({FIT_DIMENSION}):")
    asimov_data = filtered_data
    
    if FIT_DIMENSION == "1D":
        asimov_hist = {key: np.histogram(asimov_data[key][0], BINNING)[0] 
                      for key in filtered_data}
    else:
        asimov_hist = {key: np.histogram2d(asimov_data[key][0], asimov_data[key][1], BINNING)[0] 
                      for key in filtered_data}
    
    for key in asimov_hist:
        print(f"  {key}: {np.sum(asimov_hist[key]):.0f} events in histogram")
    
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
        print(f"\n{'='*60}")
        print(f"EXPOSURE: {years} years")
        print(f"{'='*60}")
        
        # Normalize and scale Asimov (do this once)
        asimov_normalized, asimov_scaled = normalize_and_scale_asimov(asimov_hist, years, filtered_rates)
        
        # Make interpolated PDFs (do this once)
        norm_pdf_luts, pdfs, bin_centers = make_normalized_interpolated_pdf(asimov_normalized)
        
        # Calculate total expected events (needed for fitting)
        total_events = sum(filtered_rates[ch] for ch in filtered_rates.keys())
        print(f"\n  Total expected events (filtered): {total_events:,.1f}/year")
        
        # Process toys ONE BY ONE to save memory
        print(f"\nGenerating and fitting {N_TOYS} toy datasets (one at a time to save memory)...")
        fit_results = []
        
        for toy_idx in range(N_TOYS):
            # Generate ONE toy dataset with Poisson + flux errors
            toy_datasets = make_toy_datasets_with_poisson_and_flux(
                filtered_data, filtered_rates, years, ngroups=1
            )
            
            # Make histogram for this toy
            if FIT_DIMENSION == "1D":
                toy_hist = {key: np.histogram(toy_datasets[key][0][0], BINNING)[0] 
                           for key in filtered_data}
            else:
                toy_hist = {key: np.histogram2d(toy_datasets[key][0][0], toy_datasets[key][0][1], BINNING)[0] 
                           for key in filtered_data}
            
            # Sum all channels to make total data histogram for fitting
            fit_data = np.sum([toy_hist[ch] for ch in filtered_data.keys()], axis=0)
            
            # Print first toy as example
            if toy_idx == 0:
                print(f"\n  Example toy 0:")
                for ch in filtered_data.keys():
                    print(f"    {ch}: {np.sum(toy_hist[ch]):.0f} events")
            
            # Fit this toy
            try:
                m = fit_with_extended_binned_nll(fit_data, channels, norm_pdf_luts, years, total_events, debug=(toy_idx==0))
                fit_results.append(m)
                
                if toy_idx == 0:
                    print(f"\n  Example fit result:")
                    for ch in channels:
                        print(f"    {ch}: {m.values[ch]:.1f} ± {m.errors[ch]:.1f}")
            except Exception as e:
                if toy_idx == 0:
                    print(f"  ERROR in fit {toy_idx+1}:")
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"  WARNING: Fit {toy_idx+1} failed: {e}")
                continue
            
            # FREE MEMORY - delete large objects after each fit
            del toy_datasets
            del toy_hist
            del fit_data
            
            # Explicit garbage collection every 50 toys to be extra safe
            if (toy_idx + 1) % 50 == 0:
                gc.collect()
            
            # Print progress every 100 toys (helpful for large N_TOYS)
            if (toy_idx + 1) % 100 == 0:
                print(f"  Progress: {toy_idx + 1}/{N_TOYS} toys completed...")
        
        print(f"  Finished fitting all {N_TOYS} toys!")
        
        # Store results 
        if len(fit_results) == 0:
            print(f"\n  ERROR: All fits failed for {years} year exposure!")
            continue
        
        ch_idx = channels.index(signal_channel)
        channel_mapping = {
            'nES': 'eES',
            'nNeutrons': 'neutrons',
            'nCosmics': 'cosmics',
            'nO16': 'nueO16',
            'nGa71': 'nueGa71'
        }
        true_val = filtered_rates[channel_mapping[signal_channel]]*years
        
        for i, m in enumerate(fit_results):
            all_results[years].append({
                'fitted': m.values[signal_channel],
                'error': m.errors[signal_channel],
                'valid': m.valid,
                'true_value': true_val
            })
        
        # Print summary
        signal_vals = [m.values[signal_channel] for m in fit_results if m.valid]
        signal_err  = [m.errors[signal_channel] for m in fit_results if m.valid]
        
        if len(signal_vals) > 0:
            mean_val = np.mean(signal_vals)
            mean_err = np.mean(signal_err)
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Load channels
    print("\n" + "=" * 80)
    print(f"LOADING DATA: {DETECTOR_NAME}_{BEAM_POWER}MW_{SHIELDING}")
    print("=" * 80)
    
    channel_cache = {}
    channels_to_load = ['nueGa71', 'eES', 'cosmics', 'nueO16'] if FIT_SCENARIO == "gallium" else ['eES', 'cosmics', 'nueO16']

    for channel_name in channels_to_load:
        print(f"  {channel_name}:")
        try:
            data = load_preprocessed_channel(DETECTOR_NAME, channel_name)
            channel_cache[channel_name] = data
        except FileNotFoundError as e:
            print(f"    ERROR: {e}")
            continue

    # Run analysis
    results = run_scenario_analysis(
        channel_cache, SHIELDING, BEAM_POWER, DETECTOR_NAME
    )

    if results is not None:
        results_dict, signal_channel = results
        
        # Save results to pickle file
        config_id = f"{DETECTOR_NAME}_{SHIELDING}_{BEAM_POWER}MW_{FIT_SCENARIO}_{FIT_DIMENSION}"
        output_file = os.path.join(RESULTS_DIR, f"results_{config_id}.pkl")
        
        output_data = {
            'config': {
                'detector': DETECTOR_NAME,
                'shielding': SHIELDING,
                'beam_power': BEAM_POWER,
                'fit_scenario': FIT_SCENARIO,
                'fit_dimension': FIT_DIMENSION
            },
            'signal_channel': signal_channel,
            'exposure_times': EXPOSURE_TIMES,
            'results': results_dict
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {output_file}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("ANALYSIS FAILED - NO RESULTS TO SAVE")
        print("=" * 80)
        sys.exit(1)
