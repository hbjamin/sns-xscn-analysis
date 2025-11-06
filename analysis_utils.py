#!/usr/bin/env python

import numpy as np
from iminuit import cost, Minuit
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter  # NEW: Added for Gaussian smoothing
from scipy.optimize import curve_fit
import os

import config as cfg

def smooth_pdf_gaussian(histogram, bin_centers, params):
    """
    Smooth histogram using Gaussian filter.
    Works for both 1D and 2D histograms - the best general-purpose smoothing method.
    
    Parameters
    ----------
    histogram : array
        Histogram values (1D or 2D array)
    bin_centers : array or tuple
        Bin center positions (not used but kept for API consistency)
    params : dict
        Parameters for Gaussian filter. Should contain 'sigma' key.
        For 1D: sigma can be a float
        For 2D: sigma can be a float (same for both dims) or [sigma_x, sigma_y]
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram (same shape as input)
    """
    sigma = params.get('sigma', 1.0)
    
    try:
        # gaussian_filter works for any dimensionality
        smooth_pdf = gaussian_filter(histogram, sigma=sigma, mode='reflect')
        # force non-negative (Gaussian smoothing can occasionally produce tiny negative values)
        smooth_pdf = np.maximum(smooth_pdf, 0)
        return smooth_pdf
    except (ValueError, RuntimeError) as e:
        print(f"    warning: gaussian smoothing failed ({e}), using original")
        return histogram

def smooth_pdf_kde(histogram, bin_centers, params):
    """
    Smooth histogram using kernel density estimation.
    Good for general shapes, automatic bandwidth selection.
    ONLY WORKS FOR 1D HISTOGRAMS.
    
    Parameters
    ----------
    histogram : array
        Histogram values (counts per bin)
    bin_centers : array
        Bin center positions
    params : dict
        Parameters for KDE (should contain 'bandwidth' key)
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram
    """
    bandwidth = params.get('bandwidth', 'scott')
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
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    warning: kde smoothing failed ({e}), using original")
        return histogram

def smooth_pdf_spline(histogram, bin_centers, params):
    """
    Smooth histogram using spline interpolation.
    Good for smooth curves, tunable smoothness parameter.
    ONLY WORKS FOR 1D HISTOGRAMS.
    
    Parameters
    ----------
    histogram : array
        Histogram values
    bin_centers : array
        Bin center positions
    params : dict
        Parameters for spline (should contain 'smoothness' key)
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram
    """
    smoothness = params.get('smoothness', 1e-3)
    try:
        spline = UnivariateSpline(bin_centers, histogram, s=smoothness)
        smooth_pdf = spline(bin_centers)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except (ValueError, TypeError) as e:
        print(f"    warning: spline smoothing failed ({e}), using original")
        return histogram

def smooth_pdf_savgol(histogram, bin_centers, params):
    """
    Smooth histogram using Savitzky-Golay filter.
    Preserves peak positions well, good for spectroscopy data.
    ONLY WORKS FOR 1D HISTOGRAMS.
    
    Parameters
    ----------
    histogram : array
        Histogram values
    bin_centers : array
        Bin center positions (not used, but kept for consistency)
    params : dict
        Parameters for Savitzky-Golay filter (window, polyorder)
    
    Returns
    -------
    smooth_pdf : array
        Smoothed histogram
    """
    window = params.get('window', 11)
    polyorder = params.get('polyorder', 3)
    if len(histogram) < window:
        window = len(histogram) if len(histogram) % 2 == 1 else len(histogram) - 1
        if window < polyorder + 2:
            print("    warning: not enough bins for savgol filter")
            return histogram
    
    try:
        smooth_pdf = savgol_filter(histogram, window, polyorder)
        smooth_pdf = np.maximum(smooth_pdf, 0)  # force non-negative
        return smooth_pdf
    except (ValueError, TypeError) as e:
        print(f"    warning: savgol smoothing failed ({e}), using original")
        return histogram

def smooth_pdf_exponential(histogram, bin_centers, params=None):
    """
    Smooth histogram by fitting exponential decay.
    Good for falling spectra - physics-motivated for backgrounds.
    ONLY WORKS FOR 1D HISTOGRAMS.
    
    Parameters
    ----------
    histogram : array
        Histogram values
    bin_centers : array
        Bin center positions
    params : dict, optional
        Not used, but kept for consistency with other methods
    
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
    except (ValueError, RuntimeError) as e:
        print(f"    warning: exponential fit failed ({e}), using original")
        return histogram

def smooth_asimov_histogram(histogram, method, params, bin_centers):
    """
    Smooth an asimov histogram using the specified method.
    
    This function applies smoothing directly to histogram bin contents,
    NOT at the event level. This is the statistically correct approach:
    the asimov histogram is already created from a limited number of events,
    and smoothing just interpolates between bins without introducing
    artificial correlations.
    
    Parameters
    ----------
    histogram : array
        Raw histogram from asimov pool (1D or 2D)
    method : str
        Smoothing method: 'gaussian', 'spline', 'kde', 'savgol', or 'exponential'
        Note: Only 'gaussian' works for 2D histograms
    params : dict
        Dictionary of parameters for all methods
    bin_centers : array or tuple
        Bin center positions (for 1D: array, for 2D: tuple of arrays)
    
    Returns
    -------
    smoothed_histogram : array
        Smoothed histogram (same shape as input)
    """
    # Get method-specific parameters
    if method in params:
        method_params = params[method]
    else:
        print(f"    warning: no parameters found for method '{method}', using defaults")
        method_params = {}
    
    # Check dimensionality
    ndim = histogram.ndim
    
    # For 2D histograms, only Gaussian smoothing is supported
    if ndim == 2 and method != 'gaussian':
        print(f"    warning: method '{method}' only works for 1D histograms")
        print(f"    falling back to 'gaussian' smoothing for 2D histogram")
        method = 'gaussian'
        method_params = params.get('gaussian', {'sigma': 1.0})
    
    # Apply smoothing based on method
    if method == 'gaussian':
        smoothed = smooth_pdf_gaussian(histogram, bin_centers, method_params)
    elif method == 'spline':
        smoothed = smooth_pdf_spline(histogram, bin_centers, method_params)
    elif method == 'kde':
        smoothed = smooth_pdf_kde(histogram, bin_centers, method_params)
    elif method == 'savgol':
        smoothed = smooth_pdf_savgol(histogram, bin_centers, method_params)
    elif method == 'exponential':
        smoothed = smooth_pdf_exponential(histogram, bin_centers, method_params)
    else:
        print(f"    warning: unknown smoothing method '{method}', using original")
        smoothed = histogram
    
    return smoothed

def resample_from_smoothed_histogram(energy, histogram, bin_edges, smoothed_histogram):
    """
    Resample energy values from a smoothed histogram using inverse transform sampling.
    
    NOTE: This function is DEPRECATED for the current analysis workflow.
    We now smooth histograms directly instead of resampling events.
    This function is kept for potential future use or legacy compatibility.
    
    Parameters
    ----------
    energy : array
        Original energy values
    histogram : array
        Original histogram (not used, kept for reference)
    bin_edges : array
        Bin edges used to create histogram
    smoothed_histogram : array
        Smoothed histogram values
    
    Returns
    -------
    resampled_energy : array
        New energy values sampled from smoothed distribution
    """
    # Normalize smoothed histogram to PDF
    bin_widths = np.diff(bin_edges)
    pdf = smoothed_histogram / (np.sum(smoothed_histogram * bin_widths))
    
    # Create CDF using cumulative sum
    cdf_values = np.concatenate([[0], np.cumsum(pdf * bin_widths)])
    cdf_values /= cdf_values[-1]  # Ensure normalized to 1
    
    # Create inverse CDF interpolator
    inverse_cdf = interp1d(cdf_values, bin_edges, 
                          kind='linear', 
                          bounds_error=False, 
                          fill_value=(bin_edges[0], bin_edges[-1]))
    
    # Sample uniform random numbers and transform
    n_samples = len(energy)
    uniform_samples = np.random.uniform(0, 1, n_samples)
    resampled_energy = inverse_cdf(uniform_samples)
    
    return resampled_energy

def smooth_energy_direction_data(data, method, params):
    """
    DEPRECATED: This function smooths at event level and is no longer used.
    
    We now smooth histograms directly (after splitting into asimov/toy pools)
    rather than resampling events from smoothed distributions.
    This function is kept for legacy compatibility only.
    
    The correct workflow is:
    1. Split raw events into asimov and toy pools
    2. Create histograms from raw events
    3. Smooth only the asimov histogram (not events, not toys)
    
    Parameters
    ----------
    data : tuple
        Tuple containing (energy, direction, ntrig, nsim)
    method : str
        Smoothing method: 'spline', 'kde', 'savgol', or 'exponential'
    params : dict
        Dictionary of parameters for all methods (will extract params for specified method)
    
    Returns
    -------
    smoothed_data : tuple
        Tuple containing (smoothed_energy, direction, ntrig, nsim)
    """
    print("    WARNING: smooth_energy_direction_data is deprecated!")
    print("    This function smooths at event level which is statistically incorrect.")
    print("    Use smooth_asimov_histogram instead to smooth histograms after splitting.")
    
    energy, direction, ntrig, nsim = data
    
    # Get method-specific parameters
    if method in params:
        method_params = params[method]
    else:
        print(f"    warning: no parameters found for method '{method}', using defaults")
        method_params = {}
    
    # Create histogram of energy
    bin_edges = cfg.ENERGY_BINS
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    hist, _ = np.histogram(energy, bins=bin_edges)
    
    # Apply smoothing
    if method == 'spline':
        smoothed_hist = smooth_pdf_spline(hist, bin_centers, method_params)
    elif method == 'kde':
        smoothed_hist = smooth_pdf_kde(hist, bin_centers, method_params)
    elif method == 'savgol':
        smoothed_hist = smooth_pdf_savgol(hist, bin_centers, method_params)
    elif method == 'exponential':
        smoothed_hist = smooth_pdf_exponential(hist, bin_centers, method_params)
    else:
        print(f"    warning: unknown smoothing method '{method}', using original")
        smoothed_hist = hist
    
    # Resample energy from smoothed distribution
    smoothed_energy = resample_from_smoothed_histogram(energy, hist, bin_edges, smoothed_hist)
    
    # Keep direction unchanged
    return smoothed_energy, direction, ntrig, nsim

def load_preprocessed_channel(detector_name, channel_name):
    """
    Load preprocessed channel data from .npz file.
    
    Parameters
    ----------
    detector_name : str
        Detector name (e.g., 'water', '1wbls')
    channel_name : str
        Channel name (e.g., 'eES', 'cosmics')
    
    Returns
    -------
    energy : array
        Reconstructed energy values
    direction : array
        Reconstructed direction values
    ntrig : int
        Number of triggered events
    nsim : int
        Number of simulated events
    """
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
    """
    Load neutron data and apply spectrum weighting only.
    Year scaling happens later when generating toys.
    
    Parameters
    ----------
    detector_name : str
        Detector name
    shielding : str
        Shielding configuration (e.g., '0ft', '1ft', '3ft')
    
    Returns
    -------
    energy_weighted : array
        Spectrum-weighted energy values
    direction_weighted : array
        Corresponding direction values
    ntrig : int
        Number of triggered events
    nsim : int
        Number of simulated neutrons
    """
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
    """
    Split data into non-overlapping sets for asimov pdf and toy sampling.
    
    Parameters
    ----------
    energy_direction_data : dict
        Dictionary with keys as channel names and values as (energy, direction) tuples
    asimov_fraction : float
        Fraction of data to use for asimov (e.g., 0.5 = 50%)
    
    Returns
    -------
    asimov_data : dict
        Data for asimov histograms
    toy_data : dict
        Data for toy sampling
    """
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
        
        print(f"  {key}: {n_total:,} → {len(asimov_idx):,} asimov + {len(toy_idx):,} toy")
    
    return asimov_data, toy_data

def filter_data_to_analysis_range(energy_direction_data, event_rates_total, 
                                  energy_min=None, energy_max=None):
    """
    Filter event data to specified energy range and calculate filtered rates.
    
    Parameters
    ----------
    energy_direction_data : dict
        Dictionary with channel data (energy, direction, ntrig, nsim)
    event_rates_total : dict
        Dictionary of total event rates per year (before filtering)
    energy_min : float, optional
        Minimum energy (defaults to cfg.ENERGY_MIN)
    energy_max : float, optional
        Maximum energy (defaults to cfg.ENERGY_MAX)
    
    Returns
    -------
    filtered_data : dict
        Filtered data as (energy, direction) tuples
    filtered_rates : dict
        Event rates after filtering
    neutron_metadata : dict
        Metadata for neutron rate calculation
    """
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
            # for neutrons, store metadata for later scaling
            filtered_rate = 0  # will be calculated later
            if key == 'neutrons':
                neutron_metadata = {
                    'nsim': nsim,
                    'filtered_count': filtered_events,
                    'total_count': total_events
                }
        
        filtered_data[key] = (energy_filtered, direction_filtered)
        filtered_rates[key] = filtered_rate
        
        print(f"  {key}: {total_events:,} → {filtered_events:,} events "
              f"({100*fraction:.1f}%), rate = {filtered_rate:.1f}/year")
    
    return filtered_data, filtered_rates, neutron_metadata

def calculate_neutron_rate_per_year(neutron_metadata, neutrons_per_mw):
    """
    Calculate expected neutron event rate per year based on simulation parameters.
    
    Parameters
    ----------
    neutron_metadata : dict
        Contains 'nsim' and 'filtered_count' from filtered neutron data
    neutrons_per_mw : float
        Expected neutrons per MW per hour per m^2
    
    Returns
    -------
    rate_per_year : float
        Expected neutron events per year (for filtered energy range)
    """
    if not neutron_metadata:
        print("    warning: no neutron metadata available")
        return 0
    
    nsim = neutron_metadata['nsim']
    filtered_count = neutron_metadata['filtered_count']
    
    # Expected neutrons per year (total) hitting the detector area
    expected_total_per_year = neutrons_per_mw * cfg.SNS_HOURS_PER_YEAR * cfg.SNS_BEAM_MW
    
    # Simulation represents nsim neutrons over NEUTRON_SIM_AREA_M2
    # Scale to one year of running
    scale_factor = expected_total_per_year / nsim
    
    # Apply to filtered events
    rate_per_year = filtered_count * scale_factor
    
    return rate_per_year

def make_toy_datasets_with_poisson_and_flux(toy_data_pool, filtered_rates, years, 
                                           ngroups, neutron_metadata=None, 
                                           neutrons_per_mw=None, flux_err=None):
    """
    Generate toy datasets by sampling from toy pool with Poisson fluctuations.
    
    Parameters
    ----------
    toy_data_pool : dict
        Dictionary of (energy, direction) arrays for each channel
    filtered_rates : dict
        Expected event rates per year
    years : float
        Exposure time in years
    ngroups : int
        Number of toy datasets to generate
    neutron_metadata : dict, optional
        Metadata for neutron rate calculation
    neutrons_per_mw : float, optional
        Neutrons per MW for neutron rate calculation
    flux_err : dict, optional
        Flux uncertainties by channel (defaults to cfg.FLUX_ERR)
    
    Returns
    -------
    toy_datasets : dict
        Dictionary with lists of [energy, direction] arrays for each channel
    """
    if flux_err is None:
        flux_err = cfg.FLUX_ERR
    
    toy_datasets = {key: [] for key in toy_data_pool.keys()}
    
    # Calculate neutron rate once (doesn't depend on toy pool size)
    neutron_rate = 0
    if 'neutrons' in toy_data_pool and neutron_metadata and neutrons_per_mw is not None:
        neutron_rate = calculate_neutron_rate_per_year(neutron_metadata, neutrons_per_mw)
    
    for key in toy_data_pool.keys():
        energies, directions = toy_data_pool[key]
        
        # Determine base rate for this channel
        if key == 'neutrons':
            base_rate = neutron_rate
        else:
            base_rate = filtered_rates[key]
        
        for i in range(ngroups):
            # Apply flux uncertainty for neutrino channels
            if key in flux_err:
                flux_variation = np.random.normal(1.0, flux_err[key])
                expected_rate = base_rate * flux_variation
            else:
                expected_rate = base_rate
            
            # Apply Poisson fluctuation
            n_events = np.random.poisson(expected_rate * years)
            
            # Sample with replacement from toy pool
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
    """
    Normalize asimov histograms to PDFs and scale by expected rates.
    
    Parameters
    ----------
    asimov_hist : dict
        Raw histograms from asimov pool
    years : float
        Exposure time
    filtered_rates : dict
        Expected rates per year
    
    Returns
    -------
    asimov_normalized : dict
        Normalized PDFs
    asimov_scaled : dict
        Scaled to expected event counts
    """
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
    """
    Create interpolated PDF functions from normalized asimov histograms.
    
    Parameters
    ----------
    asimov_normalized : dict
        Normalized histograms
    fit_dimension : str
        '1D' or '2D'
    energy_bins : array, optional
        Energy bin edges
    direction_bins : array, optional
        Direction bin edges
    
    Returns
    -------
    norm_pdf_luts : dict
        Interpolated PDF functions
    pdfs : dict
        Bin-normalized PDFs
    bin_centers : array or list
        Bin centers (1D array or list of [energy_centers, direction_centers])
    """
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
    """
    Perform extended binned negative log-likelihood fit.
    
    Parameters
    ----------
    fit_data : array
        Binned data to fit
    channels_to_fit : list
        List of channel names to include in fit
    norm_pdf_luts : dict
        Interpolated PDF functions
    years : float
        Exposure time
    total_events : float
        Total expected events
    fit_scenario : str
        'oxygen' or 'gallium'
    fit_dimension : str
        '1D' or '2D'
    binning : array or list
        Bin edges
    filtered_rates : dict
        Expected rates per year
    
    Returns
    -------
    m : Minuit
        Fitted Minuit object
    """
    
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
    except (RuntimeError, ValueError):
        pass  # fall back to hesse errors
    
    return m
