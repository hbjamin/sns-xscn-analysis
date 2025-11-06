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
                         fit_dimension, output_path):
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
    """
    if len(all_results) == 0:
        print("error: no results to plot!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # plot each config
    for idx, (config_name, result_data) in enumerate(sorted(all_results.items())):
        
        minuit_precisions = []
        bias_corrected_precisions = []
        
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
                    
                    # bias-corrected rms
                    bias = avg_fitted - true_val
                    corrected_rms = np.sqrt(np.mean([(v - bias - true_val)**2 for v in fitted_vals]))
                    bias_corr_precision = 100 * corrected_rms / true_val
                    bias_corrected_precisions.append(bias_corr_precision)
                else:
                    minuit_precisions.append(np.nan)
                    bias_corrected_precisions.append(np.nan)
            else:
                minuit_precisions.append(np.nan)
                bias_corrected_precisions.append(np.nan)
        
        # plot lines
        color = list(cfg.CHANNEL_COLORS.values())[idx % len(cfg.CHANNEL_COLORS)]
        
        # bias-corrected rms (solid line)
        ax.plot(exposure_times, bias_corrected_precisions, 
                linestyle='-', 
                label=f"{config_name} (bias-corr rms)", 
                linewidth=2, color=color)
        
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
                         fit_dimension, output_path):
    """
    Plot bias curves for exposure time
    """
    
    if len(all_results) == 0:
        print("error: no results to plot!")
        return

    print(all_results)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # plot each config
    for idx, (config_name, result_data) in enumerate(sorted(all_results.items())):
        
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
        color = list(cfg.CHANNEL_COLORS.values())[idx % len(cfg.CHANNEL_COLORS)]
        
        # bias
        ax.plot(exposure_times, avg_bias_percentage, 
                linestyle='-', 
                label=f"{config_name}", 
                linewidth=2, color=color)
    
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
