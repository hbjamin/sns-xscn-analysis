#!/usr/bin/env python

import numpy as np
from iminuit import cost, Minuit
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os

import config as cfg

def smooth_pdf_kde(histogram, bin_centers, bandwidth='scott'):
    """
    Smooth histogram using kernel density estimation.
    Good for general shapes, automatic bandwidth selection.
    
    Parameters
    ----------
    histogram : array
        Histogram values (counts per bin)
    bin_centers : array
        Bin center positions
    bandwidth : str or float
        KDE bandwidth method ('scott', 'silverman') or explicit value
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram
    """
    # create samples by repeating bin centers
    counts = histogram.astype(int)
    if np.sum(counts) == 0:
        return histogram
    samples = np.repeat(bin_centers, counts)
    
    try:
        kde = gaussian_kde(samples, bw_method=bandwidth)
        smooth_pdf = kde(bin_centers)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: kde smoothing failed, using original")
        return histogram

def smooth_pdf_spline(histogram, bin_centers, smoothness=1e-3):
    """
    Smooth histogram using spline interpolation.
    Good for smooth curves, tunable smoothness parameter.
    
    Parameters
    ----------
    histogram : array
        Histogram values
    bin_centers : array
        Bin center positions
    smoothness : float
        Spline smoothness parameter (lower = more wiggly)
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram
    """
    try:
        spline = UnivariateSpline(bin_centers, histogram, s=smoothness)
        smooth_pdf = spline(bin_centers)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: spline smoothing failed, using original")
        return histogram

def smooth_pdf_savgol(histogram, bin_centers, window=11, polyorder=3):
    """
    Smooth histogram using Savitzky-Golay filter.
    Preserves peak positions well, good for spectroscopy data.
    
    Parameters
    ----------
    histogram : array
        Histogram values
    bin_centers : array
        Bin center positions (not used, but kept for consistency)
    window : int
        Window length (must be odd, >= polyorder + 2)
    polyorder : int
        Polynomial order (must be < window)
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram
    """
    if len(histogram) < window:
        window = len(histogram) if len(histogram) % 2 == 1 else len(histogram) - 1
        if window < polyorder + 2:
            print("    warning: not enough bins for savgol filter")
            return histogram
    
    try:
        smooth_pdf = savgol_filter(histogram, window, polyorder)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: savgol smoothing failed, using original")
        return histogram

def smooth_pdf_exponential(histogram, bin_centers):
    """
    Smooth histogram by fitting exponential decay.
    Good for falling spectra - physics-motivated for backgrounds.
    
    Parameters
    ----------
    histogram : array
        Histogram values
    bin_centers : array
        Bin center positions
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram (exponential fit)
    """
    def exp_func(x, A, k, C):
        return A * np.exp(-k * x) + C
    
    # fit to non-zero bins only
    mask = histogram > 0
    if np.sum(mask) < 3:  # need at least 3 points
        print("    warning: not enough non-zero bins for exponential fit")
        return histogram
    
    try:
        popt, _ = curve_fit(exp_func, bin_centers[mask], histogram[mask],
                           p0=[np.max(histogram), 0.01, 0],
                           maxfev=5000)
        smooth_pdf = exp_func(bin_centers, *popt)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: exponential fit failed, using original")
        return histogram

def smooth_asimov_histograms(asimov_hist, fit_dimension):
    """
    Apply optional smoothing to asimov histograms.
    Smoothing happens before normalization and plotting.
    
    Parameters
    ----------
    asimov_hist : dict
        Dictionary of histograms by channel name
    fit_dimension : str
        '1D' or '2D' - determines smoothing approach
    
    Returns
    -------
    smoothed_hist : dict
        Dictionary of smoothed histograms
    """
    if not cfg.SMOOTH_ASIMOV['enabled']:
        return asimov_hist
    
    print("\nsmoothing asimov histograms:")
    method = cfg.SMOOTH_ASIMOV['method']
    params = cfg.SMOOTH_ASIMOV['params'][method]
    
    smoothed_hist = {}
    
    for ch_name, hist in asimov_hist.items():
        if ch_name not in cfg.SMOOTH_ASIMOV['channels']:
            # don't smooth this channel
            smoothed_hist[ch_name] = hist
            continue
        
        print(f"  smoothing {ch_name} with {method}...")
        
        if fit_dimension == "1D":
            # smooth 1d histogram
            bin_centers = 0.5 * (cfg.ENERGY_BINS[1:] + cfg.ENERGY_BINS[:-1])
            
            if method == 'spline':
                smooth = smooth_pdf_spline(hist, bin_centers, **params)
            elif method == 'kde':
                smooth = smooth_pdf_kde(hist, bin_centers, **params)
            elif method == 'savgol':
                smooth = smooth_pdf_savgol(hist, bin_centers, **params)
            elif method == 'exponential':
                smooth = smooth_pdf_exponential(hist, bin_centers)
            else:
                print(f"    unknown method: {method}, skipping")
                smooth = hist
            
            smoothed_hist[ch_name] = smooth
            
        else:
            # smooth 2d histogram - apply to energy projection
            print(f"    (smoothing energy projection only for 2d)")
            energy_proj = np.sum(hist, axis=1)
            bin_centers = 0.5 * (cfg.ENERGY_BINS[1:] + cfg.ENERGY_BINS[:-1])
            
            if method == 'spline':
                smooth_proj = smooth_pdf_spline(energy_proj, bin_centers, **params)
            elif method == 'kde':
                smooth_proj = smooth_pdf_kde(energy_proj, bin_centers, **params)
            elif method == 'savgol':
                smooth_proj = smooth_pdf_savgol(energy_proj, bin_centers, **params)
            elif method == 'exponential':
                smooth_proj = smooth_pdf_exponential(energy_proj, bin_centers)
            else:
                smooth_proj = energy_proj
            
            # rescale 2d histogram to match smoothed projection
            original_proj = np.sum(hist, axis=1)
            scale_factors = np.divide(smooth_proj, original_proj, 
                                     out=np.ones_like(smooth_proj), 
                                     where=original_proj!=0)
            
            smoothed_2d = hist * scale_factors[:, np.newaxis]
            smoothed_hist[ch_name] = smoothed_2d
    
    return smoothed_hist

def smooth_pdf_kde(histogram, bin_centers, bandwidth='scott'):
    # smooth using kernel density estimation
    # good for general shapes, automatic bandwidth selection
    
    # create samples by repeating bin centers
    counts = histogram.astype(int)
    if np.sum(counts) == 0:
        return histogram
    samples = np.repeat(bin_centers, counts)
    
    try:
        kde = gaussian_kde(samples, bw_method=bandwidth)
        smooth_pdf = kde(bin_centers)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: kde smoothing failed, using original")
        return histogram

def smooth_pdf_spline(histogram, bin_centers, smoothness=1e-3):
    # smooth using spline interpolation
    # good for smooth curves, tunable smoothness
    
    try:
        spline = UnivariateSpline(bin_centers, histogram, s=smoothness)
        smooth_pdf = spline(bin_centers)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: spline smoothing failed, using original")
        return histogram

def smooth_pdf_savgol(histogram, bin_centers, window=11, polyorder=3):
    # smooth using savitzky-golay filter
    # preserves peak positions
    
    if len(histogram) < window:
        window = len(histogram) if len(histogram) % 2 == 1 else len(histogram) - 1
        if window < polyorder + 2:
            print("    warning: not enough bins for savgol filter")
            return histogram
    
    try:
        smooth_pdf = savgol_filter(histogram, window, polyorder)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: savgol smoothing failed, using original")
        return histogram

def smooth_pdf_exponential(histogram, bin_centers):
    # fit exponential decay
    # good for falling spectra (physics-motivated)
    
    def exp_func(x, A, k, C):
        return A * np.exp(-k * x) + C
    
    # fit to non-zero bins only
    mask = histogram > 0
    if np.sum(mask) < 3:  # need at least 3 points
        print("    warning: not enough non-zero bins for exponential fit")
        return histogram
    
    try:
        popt, _ = curve_fit(exp_func, bin_centers[mask], histogram[mask],
                           p0=[np.max(histogram), 0.01, 0],
                           maxfev=5000)
        smooth_pdf = exp_func(bin_centers, *popt)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except:
        print("    warning: exponential fit failed, using original")
        return histogram

def smooth_asimov_histograms(asimov_hist, fit_dimension):
    # apply optional smoothing to asimov histograms
    # happens before normalization and plotting
    
    if not cfg.SMOOTH_ASIMOV['enabled']:
        return asimov_hist
    
    print("\nsmoothing asimov histograms:")
    method = cfg.SMOOTH_ASIMOV['method']
    params = cfg.SMOOTH_ASIMOV['params'][method]
    
    smoothed_hist = {}
    
    for ch_name, hist in asimov_hist.items():
        if ch_name not in cfg.SMOOTH_ASIMOV['channels']:
            # don't smooth this channel
            smoothed_hist[ch_name] = hist
            continue
        
        print(f"  smoothing {ch_name} with {method}...")
        
        if fit_dimension == "1D":
            # smooth 1d histogram
            bin_centers = 0.5 * (cfg.ENERGY_BINS[1:] + cfg.ENERGY_BINS[:-1])
            
            if method == 'spline':
                smooth = smooth_pdf_spline(hist, bin_centers, **params)
            elif method == 'kde':
                smooth = smooth_pdf_kde(hist, bin_centers, **params)
            elif method == 'savgol':
                smooth = smooth_pdf_savgol(hist, bin_centers, **params)
            elif method == 'exponential':
                smooth = smooth_pdf_exponential(hist, bin_centers)
            else:
                print(f"    unknown method: {method}, skipping")
                smooth = hist
            
            smoothed_hist[ch_name] = smooth
            
        else:
            # smooth 2d histogram - apply to energy projection
            print(f"    (smoothing energy projection only for 2d)")
            energy_proj = np.sum(hist, axis=1)
            bin_centers = 0.5 * (cfg.ENERGY_BINS[1:] + cfg.ENERGY_BINS[:-1])
            
            if method == 'spline':
                smooth_proj = smooth_pdf_spline(energy_proj, bin_centers, **params)
            elif method == 'kde':
                smooth_proj = smooth_pdf_kde(energy_proj, bin_centers, **params)
            elif method == 'savgol':
                smooth_proj = smooth_pdf_savgol(energy_proj, bin_centers, **params)
            elif method == 'exponential':
                smooth_proj = smooth_pdf_exponential(energy_proj, bin_centers)
            else:
                smooth_proj = energy_proj
            
            # rescale 2d histogram to match smoothed projection
            original_proj = np.sum(hist, axis=1)
            scale_factors = np.divide(smooth_proj, original_proj, 
                                     out=np.ones_like(smooth_proj), 
                                     where=original_proj!=0)
            
            smoothed_2d = hist * scale_factors[:, np.newaxis]
            smoothed_hist[ch_name] = smoothed_2d
    
    return smoothed_hist

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

def load_and_spectrum_weight_neutrons(detector_name, shielding):
    # load neutron data and apply spectrum weighting only
    # year scaling happens later when generating toys
    
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
    
    # apply spectrum weighting (no year scaling yet!)
    if not cfg.NEUTRON_SPECTRUM_FILE.exists():
        print(f"    warning: spectrum file not found, using uniform weighting")
        # no scaling, just use as-is
        energy_weighted = energy
        direction_weighted = direction
    else:
        # spectrum-weighted resampling
        n = np.genfromtxt(cfg.NEUTRON_SPECTRUM_FILE)
        
        # create energy bins and get expected spectrum shape
        bin_edges = np.linspace(0, 200, 201)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        n_expected_spectrum = np.interp(bin_centers, n[:, 0], n[:, 1])
        n_expected_spectrum = n_expected_spectrum / np.sum(n_expected_spectrum)  # normalize
        
        # flat simulation spectrum
        n_sim_spectrum = np.ones(len(bin_centers)) / len(bin_centers)
        
        # compute reweighting factors
        reweight = n_expected_spectrum / n_sim_spectrum
        i_reweight = np.interp(mcke, bin_centers, reweight)
        
        # resample according to spectrum (keeping roughly same total number)
        # this preserves the energy distribution shape
        probabilities = i_reweight / np.sum(i_reweight)
        N_resample = len(energy)  # keep same total number for now
        sampled_indices = np.random.choice(len(energy), size=N_resample, 
                                          p=probabilities, replace=True)
        energy_weighted = energy[sampled_indices]
        direction_weighted = direction[sampled_indices]
    
    print(f"    loaded and spectrum-weighted {len(energy_weighted):,} neutron events")
    print(f"    (year scaling will be applied during toy generation)")
    
    # return: energy, direction, and original nsim for later scaling
    return energy_weighted, direction_weighted, ntrig, nsim

def split_data_for_asimov_and_toys(energy_direction_data, asimov_fraction):
    # split data into non-overlapping sets for asimov pdf and toy sampling
    # asimov_fraction controls split (e.g. 0.5 = 50% asimov, 50% toys)
    # works with filtered data format: (energy, direction)
    
    asimov_data = {}
    toy_data = {}
    
    for key in energy_direction_data:
        energy, direction = energy_direction_data[key]
        
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
        
        print(f"  {key}: {n_total:,} â†’ {len(asimov_idx):,} asimov + {len(toy_idx):,} toy")
    
    return asimov_data, toy_data

def filter_data_to_analysis_range(energy_direction_data, event_rates_total, 
                                  energy_min=None, energy_max=None):
    if energy_min is None:
        energy_min = cfg.ENERGY_MIN
    if energy_max is None:
        energy_max = cfg.ENERGY_MAX
    
    filtered_data = {}
    filtered_rates = {}
    neutron_metadata = {}  # store neutron info for later scaling
    
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
            # for neutrons, store count and nsim for later scaling
            filtered_rate = filtered_events  # placeholder
            if key == 'neutrons':
                neutron_metadata = {
                    'nsim': nsim,
                    'filtered_count': filtered_events,
                    'total_count': total_events
                }
        
        filtered_data[key] = (energy_filtered, direction_filtered)
        filtered_rates[key] = filtered_rate
        
        print(f"  {key}: {total_events:,} â†’ {filtered_events:,} events "
              f"({100*fraction:.1f}%), rate = {filtered_rate:.1f}/year")
    
    return filtered_data, filtered_rates, neutron_metadata

def calculate_neutron_rate_for_pool(pool_size, neutron_metadata, neutrons_per_mw):
    # calculate expected neutron events per year based on pool size
    # pool_size: number of events in the toy pool after splitting
    # neutron_metadata: contains nsim and filtered_count from original data
    
    if neutron_metadata:
        nsim = neutron_metadata['nsim']
        filtered_count = neutron_metadata['filtered_count']

        print("nsim: ", nsim)
        print("filtered count: ", filtered_count)
        
        # expected neutrons per year (total)
        expected_total = neutrons_per_mw * cfg.SNS_HOURS_PER_YEAR * cfg.SNS_BEAM_MW
        
        # simulation represents nsim neutrons over NEUTRON_SIM_AREA_M2
        sim_neutrons_per_m2 = nsim / cfg.NEUTRON_SIM_AREA_M2
        
        # scale factor from simulation to one year
        scale_factor = expected_total / sim_neutrons_per_m2
        
        # rate for this pool (accounting for fraction of filtered events in pool)
        pool_fraction = pool_size / filtered_count if filtered_count > 0 else 1.0
        rate_per_year = scale_factor * pool_fraction
        
        return rate_per_year
    else:
        # fallback: use pool_size as rate
        print("falling back for neutron rate")
        return pool_size

def make_toy_datasets_with_poisson_and_flux(toy_data_pool, filtered_rates, years, 
                                           ngroups, neutron_metadata=None, 
                                           neutrons_per_mw=None, flux_err=None):
    # sample from toy_data_pool (which is separate from asimov data)
    # for neutrons, calculate expected rate based on pool size
    
    if flux_err is None:
        flux_err = cfg.FLUX_ERR
    
    toy_datasets = {key: [] for key in toy_data_pool.keys()}
    
    for key in toy_data_pool.keys():
        energies, directions = toy_data_pool[key]

        #### FIXME 
        
        # calculate expected rate for this channel
        if key == 'neutrons' and neutron_metadata and neutrons_per_mw is not None:
            print("got neutron metadata")
            # calculate neutron rate based on toy pool size
            base_rate = calculate_neutron_rate_for_pool(
                len(energies), neutron_metadata, neutrons_per_mw
            )
        else:
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
