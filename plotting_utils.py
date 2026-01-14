#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

import config as cfg

hep.style.use("ROOT")

def plot_asimov_projections(asimov_hist, years, output_path, fit_dimension):
    """
    Plot Asimov histogram projections.
    
    Parameters
    ----------
    asimov_hist : dict
        Dictionary of histograms by channel
    years : float
        Exposure time
    output_path : Path
        Where to save the plot
    fit_dimension : str
        '1D' or '2D'
    """
    if fit_dimension == "1D":
        # For 1D, just plot energy histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        en_hists = {ch_name: (hist, cfg.ENERGY_BINS) for ch_name, hist in asimov_hist.items()}
        hep.histplot(list(en_hists.values()), stack=False, histtype='step', label=en_hists.keys(), ax=ax)
        ax.semilogy()
        ax.set_xlim(cfg.ENERGY_MIN, cfg.ENERGY_MAX)
        ax.set_ylim(1,1e5)
        ax.set_ylabel(f"Events / SNS-Year / {cfg.ENERGY_BINS[1]-cfg.ENERGY_BINS[0]:.2f} MeV", fontsize=20)
        ax.set_xlabel(f"Reconstructed Electron Energy [MeV]", fontsize=20)
        ax.legend(loc='upper right', ncol=2, fontsize=20)
        ax.set_title("1D Fit (Energy Only)", fontsize=16)
    else:
        # For 2D, plot both projections
        en_hists = {ch_name: (np.sum(hist, axis=1), cfg.ENERGY_BINS) for ch_name, hist in asimov_hist.items()}
        dir_hists = {ch_name: (np.sum(hist, axis=0), cfg.DIRECTION_BINS) for ch_name, hist in asimov_hist.items()}
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        # Energy projection
        hep.histplot(list(en_hists.values()), stack=False, histtype='step', label=en_hists.keys(), ax=ax[0])
        ax[0].semilogy()
        ax[0].set_xlim(cfg.ENERGY_MIN, cfg.ENERGY_MAX)
        ax[0].set_ylim(1,1e5)
        ax[0].set_ylabel(f"Events / SNS-Year / {cfg.ENERGY_BINS[1]-cfg.ENERGY_BINS[0]:.2f} MeV", fontsize=20)
        ax[0].set_xlabel(f"Reconstructed Electron Energy [MeV]", fontsize=20)
        ax[0].legend(loc='upper right', ncol=2, fontsize=20)
        # Direction projection
        hep.histplot(list(dir_hists.values()), stack=False, histtype='step', label=dir_hists.keys(), ax=ax[1])
        ax[1].semilogy()
        ax[1].set_xlim(-1, 1)
        ax[1].set_ylabel(f"Events / SNS-Year / {cfg.DIRECTION_BINS[1]-cfg.DIRECTION_BINS[0]:.2f}", fontsize=20)
        ax[1].set_xlabel(f"Reconstructed Electron Direction [cos$\\theta$]", fontsize=20)
        ax[1].legend(loc='upper right', ncol=2, fontsize=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved asimov hists: {output_path}")
    plt.close()

def plot_asimov_and_fit_group_projections(asimov_hist, fitgroups_hist, years, output_path, 
                                         fit_dimension, n_toys_to_plot=10):
    """
    Plot Asimov and first N toy datasets overlaid.
    
    Parameters
    ----------
    asimov_hist : dict
        Asimov histograms by channel
    fitgroups_hist : list
        List of toy histograms
    years : float
        Exposure time
    output_path : Path
        Where to save the plot
    fit_dimension : str
        '1D' or '2D'
    n_toys_to_plot : int
        Number of toy datasets to overlay
    """
    if fit_dimension == "1D":
        # 1D case
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # Plot Asimov (solid lines)
        for key in asimov_hist.keys():
            hep.histplot(asimov_hist[key], bins=cfg.ENERGY_BINS, histtype='step', 
                        label=f"Asimov {key}", color=cfg.CHANNEL_COLORS.get(key, 'black'), 
                        linewidth=2, ax=ax)
        # Plot only first N toys (dashed lines)
        n_to_plot = min(n_toys_to_plot, len(fitgroups_hist))
        for group_idx in range(n_to_plot):
            fit_hist = fitgroups_hist[group_idx]
            for key in fit_hist.keys():
                # Only add label for first toy
                label = f"Toy {key}" if group_idx == 0 else None
                hep.histplot(fit_hist[key], bins=cfg.ENERGY_BINS, histtype='step', 
                            linestyle='--', color=cfg.CHANNEL_COLORS.get(key, 'black'), 
                            alpha=0.3, label=label, ax=ax)
        ax.set_xlabel("Reconstructed Energy [MeV]", fontsize=20)
        ax.semilogy()
        ax.set_ylabel("Events", fontsize=20)
        ax.set_xlim(cfg.ENERGY_MIN, cfg.ENERGY_MAX)
        ax.set_ylim(1,1e5)
        ax.legend(fontsize=12)
        ax.set_title(f"First {n_to_plot} Toys (1D Fit)", fontsize=14)
    else:
        # 2D case
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        # Plot Asimov (solid lines)
        for key in asimov_hist.keys():
            asimov_energy = np.sum(asimov_hist[key], axis=1)
            asimov_direction = np.sum(asimov_hist[key], axis=0)
            hep.histplot(asimov_energy, bins=cfg.ENERGY_BINS, histtype='step', 
                        label=f"Asimov {key}", color=cfg.CHANNEL_COLORS.get(key, 'black'), 
                        linewidth=2, ax=ax[0])
            hep.histplot(asimov_direction, bins=cfg.DIRECTION_BINS, histtype='step', 
                        label=f"Asimov {key}", color=cfg.CHANNEL_COLORS.get(key, 'black'), 
                        linewidth=2, ax=ax[1])
        # Plot only first N toys (dashed lines)
        n_to_plot = min(n_toys_to_plot, len(fitgroups_hist))
        for group_idx in range(n_to_plot):
            fit_hist = fitgroups_hist[group_idx]
            for key in fit_hist.keys():
                fit_energy = np.sum(fit_hist[key], axis=1)
                fit_direction = np.sum(fit_hist[key], axis=0)
                # Only add label for first toy
                label = f"Toy {key}" if group_idx == 0 else None
                hep.histplot(fit_energy, bins=cfg.ENERGY_BINS, histtype='step', 
                            linestyle='--', color=cfg.CHANNEL_COLORS.get(key, 'black'), 
                            alpha=0.3, label=label, ax=ax[0])
                hep.histplot(fit_direction, bins=cfg.DIRECTION_BINS, histtype='step', 
                            linestyle='--', color=cfg.CHANNEL_COLORS.get(key, 'black'), 
                            alpha=0.3, label=label, ax=ax[1])
        ax[0].set_xlabel("Reconstructed Energy [MeV]", fontsize=20)
        ax[0].semilogy()
        ax[0].set_ylabel("Events", fontsize=20)
        ax[0].set_xlim(cfg.ENERGY_MIN, cfg.ENERGY_MAX)
        ax[0].set_ylim(1,1e5)
        ax[0].legend(fontsize=12)
        ax[0].set_title(f"First {n_to_plot} Toys", fontsize=14)
        ax[1].semilogy()
        ax[1].set_xlabel("Reconstructed Direction [cos(theta)]", fontsize=20)
        ax[1].set_ylabel("Events", fontsize=20)
        ax[1].set_xlim(-1, 1)
        ax[1].legend(fontsize=12)
        ax[1].set_title(f"First {n_to_plot} Toys", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved first {n_to_plot} toy hists to: {output_path}")
    plt.close()

def plot_precision_curves(all_results, exposure_times, signal_channel,
                         fit_dimension, output_path, detector_filter=None):
    """
    Plot precision curves showing statistical uncertainty vs exposure time.
    
    Parameters
    ----------
    all_results : dict
        Results dictionary by configuration
    exposure_times : list
        Exposure times to plot
    signal_channel : str
        Signal channel name (e.g., 'nO16')
    fit_scenario : str
        Fit scenario name
    fit_dimension : str
        Fit dimension ('1D' or '2D')
    output_path : Path
        Where to save the plot
    detector_filter : str, optional
        If provided, only plot configs starting with this detector name (e.g., 'water', '1wbls')
    """
    if len(all_results) == 0:
        print("error: no results to plot!")
        return
    
    # Filter results by detector if requested
    if detector_filter is not None:
        filtered_results = {k: v for k, v in all_results.items() if k.startswith(detector_filter)}
        if len(filtered_results) == 0:
            print(f"warning: no results found for detector '{detector_filter}'")
            return
    else:
        filtered_results = all_results
    
    fig, ax = plt.subplots(figsize=(14, 9))

    tab10_colors = plt.cm.tab10.colors
    
    # plot each config
    for idx, (config_name, result_data) in enumerate(sorted(filtered_results.items())):
        
        minuit_precisions = []
        statistical_precisions = []

        print(f"Exposure times: {exposure_times}")
        
        for years in exposure_times:
            if years in result_data and len(result_data[years]) > 0:
                errors = [r['error'] for r in result_data[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in result_data[years] if r['valid']]
                
                if len(errors) > 0 and len(fitted_vals) > 0:
                    true_val = result_data[years][0]['true_value']
                    
                    # minuit statistical precision
                    avg_error = np.mean(errors)
                    avg_fitted = np.mean(fitted_vals)
                    minuit_precision = 100 * avg_error / avg_fitted
                    minuit_precisions.append(minuit_precision)

                    # statistical spread (RMS around mean)
                    rms_around_mean = np.sqrt(np.mean([(v - avg_fitted)**2 for v in fitted_vals]))
                    precision_around_mean = 100 * rms_around_mean / true_val
                    statistical_precisions.append(precision_around_mean)
                    
                else:
                    minuit_precisions.append(np.nan)
                    statistical_precisions.append(np.nan)
            else:
                minuit_precisions.append(np.nan)
                statistical_precisions.append(np.nan)
        
        # plot lines
        #color = list(cfg.CHANNEL_COLORS.values())[idx % len(cfg.CHANNEL_COLORS)]
        color = tab10_colors[idx % len(tab10_colors)]
        
        # bias-corrected rms (solid line)
        ax.plot(exposure_times, statistical_precisions, 
                linestyle='-', 
                label=f"{config_name} (rms around mean)", 
                linewidth=2, color=color)

        print(f"{config_name}: {minuit_precisions}")
        
        # minuit stat (dashed line)
        ax.plot(exposure_times, minuit_precisions, 
                linestyle='--',
                label=f"{config_name} (minuit avg err)", 
                linewidth=2, color=color)
    
    # formatting
    signal_label = cfg.SIGNAL_LABELS.get(signal_channel, signal_channel)
    ax.set_xlabel("SNS years", fontsize=16)
    ax.set_ylabel(f"Statistical precision on {signal_label} (%)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.tick_params(labelsize=14)
    
    ax.set_xlim(min(exposure_times) - 0.1, max(exposure_times) + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved precision curves: {output_path}")
    plt.close()

def plot_bias_curves(all_results, exposure_times, signal_channel,
                         fit_dimension, output_path, detector_filter=None):
    """
    Plot bias curves for exposure time.
    
    Parameters
    ----------
    all_results : dict
        Results dictionary by configuration
    exposure_times : list
        Exposure times to plot
    signal_channel : str
        Signal channel name
    fit_dimension : str
        Fit dimension ('1D' or '2D')
    output_path : Path
        Where to save the plot
    detector_filter : str, optional
        If provided, only plot configs starting with this detector name (e.g., 'water', '1wbls')
    """
    
    if len(all_results) == 0:
        print("error: no results to plot!")
        return
    
    # Filter results by detector if requested
    if detector_filter is not None:
        filtered_results = {k: v for k, v in all_results.items() if k.startswith(detector_filter)}
        if len(filtered_results) == 0:
            print(f"warning: no results found for detector '{detector_filter}'")
            return
    else:
        filtered_results = all_results
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # plot each config
    for idx, (config_name, result_data) in enumerate(sorted(filtered_results.items())):
        
        avg_bias_percentage = []
        
        for years in exposure_times:
            if years in result_data and len(result_data[years]) > 0:
                errors = [r['error'] for r in result_data[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in result_data[years] if r['valid']]
                
                if len(errors) > 0 and len(fitted_vals) > 0:
                    true_val = result_data[years][0]['true_value']
                    
                    # average bias percentage
                    avg_fitted = np.mean(fitted_vals)
                    avg_bias_percentage.append(100 * (avg_fitted - true_val) / true_val)
                    
                else:
                    avg_bias_percentage.append(np.nan)
            else:
                avg_bias_percentage.append(np.nan)
        
        # plot lines
        #color = list(cfg.CHANNEL_COLORS.values())[idx % len(cfg.CHANNEL_COLORS)]
        colors = plt.cm.tab10(np.linspace(0,1,len(filtered_results.items())))
        
        # bias
        ax.plot(exposure_times, avg_bias_percentage, 
                linestyle='-', 
                label=f"{config_name}", 
                linewidth=2, color=colors[idx])
    
    # formatting
    signal_label = cfg.SIGNAL_LABELS.get(signal_channel, signal_channel)
    ax.set_xlabel("SNS years", fontsize=16)
    ax.set_ylabel(f"Average Bias {signal_label} (%)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.tick_params(labelsize=14)
    
    ax.set_xlim(min(exposure_times) - 0.1, max(exposure_times) + 0.1)
    #ax.set_ylim(-50,50)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved precision curves: {output_path}")
    plt.close()

def plot_bias_distributions(result_data, exposure_times, config_name, signal_channel, 
                            fit_scenario, fit_dimension, output_path):
    """
    Plot histograms of bias distributions for each exposure time.
    Shows how the fitted values are distributed around the true value.
    
    Parameters
    ----------
    result_data : dict
        Results dictionary for a single configuration: {year: [fit_results]}
    exposure_times : list
        List of exposure times to plot
    config_name : str
        Configuration name (e.g., 'water_0ft_10npmw')
    signal_channel : str
        Signal channel name (e.g., 'nO16')
    fit_scenario : str
        Fit scenario ('oxygen' or 'gallium')
    fit_dimension : str
        Fit dimension ('1D' or '2D')
    output_path : Path
        Where to save the plot
    """
    if len(result_data) == 0:
        print(f"  warning: no results to plot for {config_name}")
        return
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    # colors for different exposure times
    #colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(exposure_times)))
    import matplotlib as mpl
    colors = plt.cm.Set2(np.linspace(0,1,len(exposure_times)))
    
    # determine reasonable bin range based on all data
    all_biases = []
    for years in exposure_times:
        if years in result_data and len(result_data[years]) > 0:
            fitted_vals = [r['fitted'] for r in result_data[years] if r['valid']]
            if len(fitted_vals) > 0:
                true_val = result_data[years][0]['true_value']
                biases = [v - true_val for v in fitted_vals]
                all_biases.extend(biases)
    
    if len(all_biases) == 0:
        print(f"  warning: no valid biases to plot for {config_name}")
        return
    
    # set bin range to capture most data (with some padding)
    bias_range = (np.percentile(all_biases, 1), np.percentile(all_biases, 99))
    bin_padding = (bias_range[1] - bias_range[0]) * 0.1
    #bins = np.linspace(bias_range[0] - bin_padding, bias_range[1] + bin_padding, 30)
    bins = np.linspace(-100,100,17)
    
    # plot histogram for each exposure time
    for idx, years in enumerate(exposure_times):
        if years in result_data and len(result_data[years]) > 0:
            fitted_vals = [r['fitted'] for r in result_data[years] if r['valid']]
            
            if len(fitted_vals) > 0:
                true_val = result_data[years][0]['true_value']
                biases = [100 *(v - true_val) / true_val for v in fitted_vals]
                
                # calculate statistics for legend
                mean_bias = np.mean(biases)
                std_bias = np.std(biases)
                
                # plot histogram
                #ax.hist(biases, bins=bins, alpha=0.6, 
                #       label=f'{years:.1f} yr (μ={mean_bias:.1f}, σ={std_bias:.1f})',
                #       color=colors[idx], edgecolor='black', linewidth=0.5)
                ax.hist(biases, bins=bins, histtype='step', 
                       label=f'{years:.1f} yr (μ={mean_bias:.1f}, σ={std_bias:.1f})', color=colors[idx])
    
    # add vertical line at zero bias
    #ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero bias', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    
    # formatting
    signal_label = cfg.SIGNAL_LABELS.get(signal_channel, signal_channel)
    ax.set_xlabel(f'Bias % on {signal_label} (fitted - true) / fitted', fontsize=14)
    ax.set_ylabel('Number of toy experiments', fontsize=14)
    #ax.set_title(f'Bias Distributions: {config_name}', fontsize=16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  saved bias distributions: {output_path.name}")
    plt.close()

def plot_peak_bin_precision(all_histogram_data, exposure_times, signal_channel,
                            fit_scenario, fit_dimension, output_path, detector_filter=None,
                            use_1d_energy=True):
    """
    Plot simple Poisson precision for the peak signal bin: sqrt(S+B)/S.
    This is a sanity check comparing full likelihood fit to simple counting statistics.

    Parameters
    ----------
    all_histogram_data : dict
        Dictionary with structure: {config_key: {'asimov_hist': ..., 'filtered_rates': ...}}
    exposure_times : list
        Exposure times to plot
    signal_channel : str
        Signal channel name (e.g., 'nO16')
    fit_scenario : str
        Fit scenario ('oxygen' or 'gallium')
    fit_dimension : str
        Fit dimension ('1D' or '2D')
    output_path : Path
        Where to save the plot
    detector_filter : str, optional
        If provided, only plot configs starting with this detector name
    use_1d_energy : bool, optional
        If True (default), always use 1D energy peak bin by projecting over direction.
        If False, use the native fit dimensionality (1D for 1D fits, 2D for 2D fits).
    """
    if len(all_histogram_data) == 0:
        print("error: no histogram data to plot!")
        return

    # Filter by detector if requested
    if detector_filter is not None:
        filtered_data = {k: v for k, v in all_histogram_data.items() if k.startswith(detector_filter)}
        if len(filtered_data) == 0:
            print(f"warning: no histogram data found for detector '{detector_filter}'")
            return
    else:
        filtered_data = all_histogram_data

    fig, ax = plt.subplots(figsize=(14, 9))

    # Map signal_channel to histogram key
    signal_hist_key = cfg.CHANNEL_MAPPING.get(signal_channel, signal_channel)
    if signal_hist_key.startswith('n'):
        # Remove 'n' prefix for histogram keys (nO16 -> nueO16, etc)
        if signal_channel == 'nO16':
            signal_hist_key = 'nueO16'
        elif signal_channel == 'nGa71':
            signal_hist_key = 'nueGa71'

    # Use tab10 colors explicitly
    tab10_colors = plt.cm.tab10.colors

    # Plot each config
    for idx, (config_name, hist_data) in enumerate(sorted(filtered_data.items())):
        asimov_hist = hist_data['asimov_hist']
        filtered_rates = hist_data['filtered_rates']

        if signal_hist_key not in asimov_hist:
            print(f"  warning: signal channel {signal_hist_key} not found in {config_name}")
            continue

        precisions = []

        for years in exposure_times:
            # Get raw histogram from asimov pool (these are counts, not normalized)
            signal_hist_raw = asimov_hist[signal_hist_key]

            # Decide whether to use 1D energy or native dimensionality
            if use_1d_energy or signal_hist_raw.ndim == 1:
                # Use 1D energy projection
                if signal_hist_raw.ndim == 2:
                    # Sum over direction axis to get energy projection
                    signal_hist_counts = np.sum(signal_hist_raw, axis=1)
                else:
                    signal_hist_counts = signal_hist_raw

                # Normalize to PDF then scale by expected rate
                signal_sum = np.sum(signal_hist_counts)
                if signal_sum > 0:
                    signal_hist = (signal_hist_counts / signal_sum) * filtered_rates[signal_hist_key] * years
                else:
                    precisions.append(np.nan)
                    continue

                # Find bin with maximum signal
                peak_bin_idx = np.argmax(signal_hist)
                signal_in_peak = signal_hist[peak_bin_idx]

                # Sum all backgrounds in the same bin
                background_in_peak = 0
                for key in asimov_hist.keys():
                    if key != signal_hist_key:
                        bg_hist_raw = asimov_hist[key]

                        # Project to 1D energy if needed
                        if bg_hist_raw.ndim == 2:
                            bg_hist_counts = np.sum(bg_hist_raw, axis=1)
                        else:
                            bg_hist_counts = bg_hist_raw

                        # Normalize to PDF then scale by expected rate
                        bg_sum = np.sum(bg_hist_counts)
                        if bg_sum > 0:
                            bg_hist = (bg_hist_counts / bg_sum) * filtered_rates[key] * years
                            background_in_peak += bg_hist[peak_bin_idx]

            else:  # use_1d_energy=False and 2D histogram
                # Use full 2D histogram (energy × direction)
                signal_hist_counts = signal_hist_raw

                # Normalize to PDF then scale by expected rate
                signal_sum = np.sum(signal_hist_counts)
                if signal_sum > 0:
                    signal_hist = (signal_hist_counts / signal_sum) * filtered_rates[signal_hist_key] * years
                else:
                    precisions.append(np.nan)
                    continue

                # Find bin with maximum signal in 2D
                peak_bin_idx = np.unravel_index(np.argmax(signal_hist), signal_hist.shape)
                signal_in_peak = signal_hist[peak_bin_idx]

                # Sum all backgrounds in the same 2D bin
                background_in_peak = 0
                for key in asimov_hist.keys():
                    if key != signal_hist_key:
                        bg_hist_raw = asimov_hist[key]

                        # Normalize to PDF then scale by expected rate
                        bg_sum = np.sum(bg_hist_raw)
                        if bg_sum > 0:
                            bg_hist = (bg_hist_raw / bg_sum) * filtered_rates[key] * years
                            background_in_peak += bg_hist[peak_bin_idx]

            # Debug print for first config and first year
            if idx == 0 and years == exposure_times[0]:
                mode = "1D energy" if use_1d_energy or signal_hist_raw.ndim == 1 else "2D energy×direction"
                print(f"\nDebug - Peak bin calculation ({mode}) for {config_name}, {years} years:")
                print(f"  Signal in peak: {signal_in_peak:.1f}")
                print(f"  Background in peak: {background_in_peak:.1f}")
                print(f"  Total in peak: {signal_in_peak + background_in_peak:.1f}")

            # Calculate simple Poisson precision
            total_in_peak = signal_in_peak + background_in_peak
            if signal_in_peak > 0:
                precision_percent = 100 * np.sqrt(total_in_peak) / signal_in_peak
                precisions.append(precision_percent)
            else:
                precisions.append(np.nan)

        # Plot line with tab10 colors
        color = tab10_colors[idx % len(tab10_colors)]
        ax.plot(exposure_times, precisions,
                linestyle='-', marker='o',
                label=config_name,
                linewidth=2, color=color)

    # Formatting
    signal_label = cfg.SIGNAL_LABELS.get(signal_channel, signal_channel)
    ax.set_xlabel("SNS years", fontsize=16)
    ax.set_ylabel(f"Peak Bin Precision on {signal_label}: √(S+B)/S (%)", fontsize=16)

    # Update title based on mode
    #if use_1d_energy:
    #    title = "Simple Poisson Counting Statistics (Peak Energy Bin)"
    #else:
    #    title = "Simple Poisson Counting Statistics (Peak Bin - Native Dimensionality)"
    #ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.tick_params(labelsize=14)

    ax.set_xlim(min(exposure_times) - 0.1, max(exposure_times) + 0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved peak bin precision curves: {output_path}")
    plt.close()
