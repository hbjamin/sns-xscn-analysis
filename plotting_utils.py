#!/usr/bin/env python

import numpy as np
from iminuit import cost, Minuit
from scipy.interpolate import RegularGridInterpolator, interp1d
import os

import config as cfg

def load_preprocessed_channel(detector_name, channel_name):
    filename = f"{detector_name}_{channel_name}.npz"
    filepath = cfg.PREPROCESSED_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"file not found: {filepath}")
    
    data = np.load(filepath)
    energy = data['energy']
    direction = data['direction']
    ntrig = int(data['ntrig'])
    nsim = int(data['nsim'])
    
    print(f"    {channel_name}: loaded {len(energy):,} events from {filename}")
    return energy, direction, ntrig, nsim

def load_neutrons_and_scale(detector_name, shielding, neutrons_per_mw):
    filename = f"{detector_name}_neutrons_{shielding}.npz"
    filepath = cfg.PREPROCESSED_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"file not found: {filepath}")
    
    data = np.load(filepath)
    energy = data['energy']
    direction = data['direction']
    mcke = data['mcke']
    ntrig = int(data['ntrig'])
    nsim = int(data['nsim'])
    
    # scale neutrons by spectrum
    if not cfg.NEUTRON_SPECTRUM_FILE.exists():
        print(f"    warning: spectrum file not found, using uniform scaling")
        scale_factor = neutrons_per_mw / 100.0
        N_new = int(len(energy) * scale_factor)
        sampled_indices = np.random.choice(len(energy), size=N_new, replace=True)
        energy_scaled = energy[sampled_indices]
        direction_scaled = direction[sampled_indices]
    else:
        # spectrum-weighted scaling
        n = np.genfromtxt(cfg.NEUTRON_SPECTRUM_FILE)
        
        # calculate expected neutrons per year
        sim_neutrons = nsim / cfg.NEUTRON_SIM_AREA_M2  # neutrons per m^2
        expected_neutrons = neutrons_per_mw * cfg.SNS_HOURS_PER_YEAR * cfg.SNS_BEAM_MW
        
        # create energy bins and compute expected spectrum
        bin_edges = np.linspace(0, 200, 201)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        n_expected = np.interp(bin_centers, n[:, 0], n[:, 1])
        n_expected *= expected_neutrons / np.sum(n_expected)
        
        # compute per-event scaling factors
        n_sim_per_bin = sim_neutrons / len(bin_centers)
        n_scaling = n_expected / n_sim_per_bin
        i_scaling = np.interp(mcke, bin_centers, n_scaling)
        
        # resample with scaling
        N_new = int(np.sum(i_scaling))
        probabilities = i_scaling / np.sum(i_scaling)
        sampled_indices = np.random.choice(len(energy), size=N_new, p=probabilities, replace=True)
        energy_scaled = energy[sampled_indices]
        direction_scaled = direction[sampled_indices]
    
    print(f"    loaded and scaled {len(energy_scaled):,} neutron events ({neutrons_per_mw} neutrons/mw)")
    return energy_scaled, direction_scaled, ntrig, nsim

def split_data_for_asimov_and_toys(energy_direction_data, asimov_fraction):
    # split data into non-overlapping sets for asimov pdf and toy sampling
    # asimov_fraction controls split (e.g. 0.5 = 50% asimov, 50% toys)
    
    asimov_data = {}
    toy_data = {}
    
    for key in energy_direction_data:
        energy, direction, ntrig, nsim = energy_direction_data[key]
        
        # shuffle indices
        n_total = len(energy)
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        # split
        n_asimov = int(n_total * asimov_fraction)
        asimov_idx = indices[:n_asimov]
        toy_idx = indices[n_asimov:]
        
        # store split data
        asimov_data[key] = (energy[asimov_idx], direction[asimov_idx])
        toy_data[key] = (energy[toy_idx], direction[toy_idx])
        
        print(f"  {key}: {n_total:,} → {len(asimov_idx):,} asimov + {len(toy_idx):,} toy")
    
    return asimov_data, toy_data

def filter_data_to_analysis_range(energy_direction_data, event_rates_total, 
                                  energy_min=None, energy_max=None):
    if energy_min is None:
        energy_min = cfg.ENERGY_MIN
    if energy_max is None:
        energy_max = cfg.ENERGY_MAX
    
    filtered_data = {}
    filtered_rates = {}
    
    print(f"\nfiltering data to energy range [{energy_min}, {energy_max}] mev:")
    
    for key in energy_direction_data:
        energy, direction, ntrig, nsim = energy_direction_data[key]
        
        # apply energy filter
        mask = (energy >= energy_min) & (energy <= energy_max)
        energy_filtered = energy[mask]
        direction_filtered = direction[mask]
        
        # calculate filtered rate
        total_events = len(energy)
        filtered_events = len(energy_filtered)
        fraction = filtered_events / total_events if total_events > 0 else 0
        
        if key in event_rates_total and event_rates_total[key] > 0:
            filtered_rate = event_rates_total[key] * fraction
        else:
            # for neutrons, use filtered count directly
            filtered_rate = filtered_events
        
        filtered_data[key] = (energy_filtered, direction_filtered)
        filtered_rates[key] = filtered_rate
        
        print(f"  {key}: {total_events:,} → {filtered_events:,} events "
              f"({100*fraction:.1f}%), rate = {filtered_rate:.1f}/year")
    
    return filtered_data, filtered_rates

def rescale_neutron_rate_after_split(filtered_rates, asimov_fraction):
    # after splitting data, we need to rescale neutron rate appropriately
    # each split represents full year, but with asimov_fraction of statistics
    # the filtered_rate for neutrons is based on ALL data, need to adjust
    
    if 'neutrons' in filtered_rates:
        # neutron rate should be scaled by asimov_fraction since we only used that fraction
        filtered_rates['neutrons'] = filtered_rates['neutrons'] * asimov_fraction
        print(f"  rescaled neutron rate after split: {filtered_rates['neutrons']:.1f}/year")
    
    return filtered_rates

def make_toy_datasets_with_poisson_and_flux(toy_data_pool, filtered_rates, years, 
                                           ngroups, flux_err=None):
    # sample from toy_data_pool (which is separate from asimov data)
    # sample WITH replacement from the toy pool
    
    if flux_err is None:
        flux_err = cfg.FLUX_ERR
    
    toy_datasets = {key: [] for key in toy_data_pool.keys()}
    
    for key in toy_data_pool.keys():
        energies, directions = toy_data_pool[key]
        base_rate = filtered_rates[key]
        
        for i in range(ngroups):
            # apply flux uncertainty for neutrino channels
            if key in flux_err:
                flux_variation = np.random.normal(1.0, flux_err[key])
                expected_rate = base_rate * flux_variation
            else:
                expected_rate = base_rate
            
            # apply poisson fluctuation
            n_events = np.random.poisson(expected_rate * years)
            
            # sample with replacement from toy pool
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
        # normalize to pdf
        norm = np.sum(hist)
        if norm > 0:
            asimov_normalized[ch_name] = hist / norm
        else:
            asimov_normalized[ch_name] = hist
        
        # scale by expected rate
        asimov_scaled[ch_name] = asimov_normalized[ch_name] * filtered_rates[ch_name] * years
    
    return asimov_normalized, asimov_scaled

def make_normalized_interpolated_pdf(asimov_normalized, fit_dimension, 
                                    energy_bins=None, direction_bins=None):
    if energy_bins is None:
        energy_bins = cfg.ENERGY_BINS
    if direction_bins is None:
        direction_bins = cfg.DIRECTION_BINS
    
    if fit_dimension == "1D":
        # 1d case: energy only
        bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])
        
        norm_pdf_luts = {}
        pdfs = {}
        
        for ch_name, hist in asimov_normalized.items():
            # normalize by bin width
            h = hist.copy()
            dx = np.diff(energy_bins)
            h = h / dx
            
            # create 1d interpolator
            interpolator = interp1d(
                bin_centers, h,
                bounds_error=False, fill_value=0,
                kind='linear'
            )
            
            norm_pdf_luts[ch_name] = interpolator
            pdfs[ch_name] = h
        
        return norm_pdf_luts, pdfs, bin_centers
        
    else:
        # 2d case: energy and direction
        bin_centers = [
            0.5 * (energy_bins[1:] + energy_bins[:-1]),
            0.5 * (direction_bins[1:] + direction_bins[:-1])
        ]
        
        norm_pdf_luts = {}
        pdfs = {}
        
        for ch_name, hist in asimov_normalized.items():
            # normalize by bin area
            h = hist.copy()
            dx = np.diff(energy_bins)
            dy = np.diff(direction_bins)
            
            for i in range(len(dx)):
                for j in range(len(dy)):
                    h[i, j] = h[i, j] / (dx[i] * dy[j])
            
            # create 2d interpolator
            interpolator = RegularGridInterpolator(
                (bin_centers[0], bin_centers[1]), h,
                bounds_error=False, fill_value=0
            )
            
            norm_pdf_luts[ch_name] = interpolator
            pdfs[ch_name] = h
        
        return norm_pdf_luts, pdfs, bin_centers

def fit_with_extended_binned_nll(fit_data, channels_to_fit, norm_pdf_luts, 
                                years, total_events, fit_scenario, fit_dimension,
                                binning, filtered_rates):
    
    # define pdf functions based on dimension and scenario
    if fit_dimension == "1D":
        if fit_scenario == "oxygen":
            def pdf_cut(xe, nES, nNeutrons, nCosmics, nO16):
                result = (nES * norm_pdf_luts['eES'](xe) +
                         nNeutrons * norm_pdf_luts['neutrons'](xe) +
                         nCosmics * norm_pdf_luts['cosmics'](xe) +
                         nO16 * norm_pdf_luts['nueO16'](xe))
                return result
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16):
                nll = cost.ExtendedBinnedNLL(fit_data, binning, pdf_cut, 
                                            use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16)
                total_fit = nES + nNeutrons + nCosmics + nO16
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=filtered_rates['eES']*years,
                      nNeutrons=filtered_rates['neutrons']*years,
                      nCosmics=filtered_rates['cosmics']*years,
                      nO16=filtered_rates['nueO16']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16'] = (0, None)
            
        elif fit_scenario == "gallium":
            def pdf_cut(xe, nES, nNeutrons, nCosmics, nO16, nGa71):
                result = (nES * norm_pdf_luts['eES'](xe) +
                         nNeutrons * norm_pdf_luts['neutrons'](xe) +
                         nCosmics * norm_pdf_luts['cosmics'](xe) +
                         nO16 * norm_pdf_luts['nueO16'](xe) +
                         nGa71 * norm_pdf_luts['nueGa71'](xe))
                return result
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16, nGa71):
                nll = cost.ExtendedBinnedNLL(fit_data, binning, pdf_cut,
                                            use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16, nGa71)
                total_fit = nES + nNeutrons + nCosmics + nO16 + nGa71
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=filtered_rates['eES']*years,
                      nNeutrons=filtered_rates['neutrons']*years,
                      nCosmics=filtered_rates['cosmics']*years,
                      nO16=filtered_rates['nueO16']*years,
                      nGa71=filtered_rates['nueGa71']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71'] = (0, None)
    
    else:  # 2d case
        if fit_scenario == "oxygen":
            def pdf_cut(xe_ye, nES, nNeutrons, nCosmics, nO16):
                xe, ye = xe_ye
                original_shape = xe.shape
                points = np.column_stack([xe.ravel(), ye.ravel()])
                
                result = (nES * norm_pdf_luts['eES'](points).reshape(original_shape) +
                         nNeutrons * norm_pdf_luts['neutrons'](points).reshape(original_shape) +
                         nCosmics * norm_pdf_luts['cosmics'](points).reshape(original_shape) +
                         nO16 * norm_pdf_luts['nueO16'](points).reshape(original_shape))
                return result
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16):
                nll = cost.ExtendedBinnedNLL(fit_data, binning, pdf_cut,
                                            use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16)
                total_fit = nES + nNeutrons + nCosmics + nO16
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=filtered_rates['eES']*years,
                      nNeutrons=filtered_rates['neutrons']*years,
                      nCosmics=filtered_rates['cosmics']*years,
                      nO16=filtered_rates['nueO16']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16'] = (0, None)
            
        elif fit_scenario == "gallium":
            def pdf_cut(xe_ye, nES, nNeutrons, nCosmics, nO16, nGa71):
                xe, ye = xe_ye
                original_shape = xe.shape
                points = np.column_stack([xe.ravel(), ye.ravel()])
                
                result = (nES * norm_pdf_luts['eES'](points).reshape(original_shape) +
                         nNeutrons * norm_pdf_luts['neutrons'](points).reshape(original_shape) +
                         nCosmics * norm_pdf_luts['cosmics'](points).reshape(original_shape) +
                         nO16 * norm_pdf_luts['nueO16'](points).reshape(original_shape) +
                         nGa71 * norm_pdf_luts['nueGa71'](points).reshape(original_shape))
                return result
            
            def cost_with_total_constraint(nES, nNeutrons, nCosmics, nO16, nGa71):
                nll = cost.ExtendedBinnedNLL(fit_data, binning, pdf_cut,
                                            use_pdf="approximate")(nES, nNeutrons, nCosmics, nO16, nGa71)
                total_fit = nES + nNeutrons + nCosmics + nO16 + nGa71
                penalty = ((total_fit - total_events*years)**2) / (total_events*years)
                return nll + penalty
            
            m = Minuit(cost_with_total_constraint,
                      nES=filtered_rates['eES']*years,
                      nNeutrons=filtered_rates['neutrons']*years,
                      nCosmics=filtered_rates['cosmics']*years,
                      nO16=filtered_rates['nueO16']*years,
                      nGa71=filtered_rates['nueGa71']*years)
            m.limits['nES', 'nNeutrons', 'nCosmics', 'nO16', 'nGa71'] = (0, None)
    
    # perform fit
    m.migrad()
    m.hesse()
    
    # try minos error analysis
    try:
        m.minos()
    except:
        pass  # fall back to hesse errors
    
    return m
