#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import glob

import config as cfg

hep.style.use("ROOT")

# filter settings (set to none to process all)
FILTER_FIT_SCENARIO = "oxygen"  # oxygen/gallium/none
FILTER_FIT_DIMENSION = "2D"     # 1d/2d/none

def load_all_results(results_dir, filter_scenario=None, filter_dimension=None):
    result_files = sorted(results_dir.glob("results_*.pkl"))
    
    if len(result_files) == 0:
        print(f"error: no result files found in {results_dir}")
        return {}
    
    print(f"found {len(result_files)} result files")
    
    all_results = {}
    for result_file in result_files:
        try:
            with open(result_file, 'rb') as f:
                data = pickle.load(f)
            
            config = data['config']
            
            # apply filters
            if filter_scenario is not None and config['fit_scenario'] != filter_scenario:
                continue
            if filter_dimension is not None and config['fit_dimension'] != filter_dimension:
                continue
            
            # create config key
            config_key = f"{config['detector']}_{config['shielding']}_{config['neutrons_per_mw']}npmw"
            
            # store data
            all_results[config_key] = {
                'config': config,
                'signal_channel': data['signal_channel'],
                'exposure_times': data['exposure_times'],
                'results': data['results']
            }
            
            print(f"  loaded: {config_key} ({config['fit_scenario']}, {config['fit_dimension']})")
            
        except Exception as e:
            print(f"  error loading {result_file}: {e}")
            continue
    
    return all_results

def plot_precision_curves(all_results, output_path, detector_filter=None):
    # filter results by detector type if specified
    if detector_filter:
        filtered_results = {k: v for k, v in all_results.items() 
                           if k.startswith(detector_filter)}
        if len(filtered_results) == 0:
            print(f"warning: no results found for detector type '{detector_filter}'")
            return
        plot_results = filtered_results
        detector_label = detector_filter.upper()
    else:
        plot_results = all_results
        detector_label = "all detectors"
    
    if len(plot_results) == 0:
        print("error: no results to plot!")
        return
    
    # get signal channel and exposure times from first result
    first_result = list(plot_results.values())[0]
    signal_channel = first_result['signal_channel']
    exposure_times = first_result['exposure_times']
    fit_scenario = first_result['config']['fit_scenario']
    fit_dimension = first_result['config']['fit_dimension']
    
    # create figure
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # plot each config
    for idx, (config_name, result_data) in enumerate(sorted(plot_results.items())):
        results = result_data['results']
        
        minuit_precisions = []
        bias_corrected_precisions = []
        
        for years in exposure_times:
            if years in results and len(results[years]) > 0:
                # get errors and fitted values from valid fits
                errors = [r['error'] for r in results[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in results[years] if r['valid']]
                
                if len(errors) > 0 and len(fitted_vals) > 0:
                    # get true value
                    true_val = results[years][0]['true_value']
                    
                    # calculate minuit statistical precision
                    avg_error = np.mean(errors)
                    avg_fitted = np.mean(fitted_vals)
                    minuit_precision = 100 * avg_error / avg_fitted
                    minuit_precisions.append(minuit_precision)
                    
                    # calculate bias-corrected rms
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
        
        # plot with solid lines only (no markers)
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
    ax.set_xlabel("sns years", fontsize=16)
    ax.set_ylabel(f"statistical precision on {signal_label} (%)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.tick_params(labelsize=14)
    
    # set reasonable axis limits
    ax.set_xlim(min(exposure_times) - 0.1, max(exposure_times) + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved precision curves ({detector_label}): {output_path}")
    plt.close()

def create_comparison_table(all_results):
    if len(all_results) == 0:
        print("error: no results to summarize!")
        return
    
    print("\n" + "="*140)
    print("summary table")
    print("="*140)
    
    # get exposure times from first result
    first_result = list(all_results.values())[0]
    exposure_times = first_result['exposure_times']
    signal_channel = first_result['signal_channel']
    
    # print header
    print(f"{'config':<30} {'years':<8} {'minuit stat (%)':>18} {'avg bias':>15} {'bias-corr rms (%)':>20}")
    print("-"*140)
    
    # print each config
    for config_name, result_data in sorted(all_results.items()):
        results = result_data['results']
        
        for years in exposure_times:
            if years in results and len(results[years]) > 0:
                errors = [r['error'] for r in results[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in results[years] if r['valid']]
                
                if len(errors) > 0 and len(fitted_vals) > 0:
                    # get metrics
                    avg_error = np.mean(errors)
                    avg_fitted = np.mean(fitted_vals)
                    true_val = results[years][0]['true_value']
                    
                    # calculate bias
                    bias = avg_fitted - true_val
                    
                    # calculate bias-corrected rms around truth
                    corrected_rms = np.sqrt(np.mean([(v - bias - true_val)**2 for v in fitted_vals]))
                    
                    # minuit statistical precision
                    minuit_stat_precision = 100 * avg_error / avg_fitted
                    
                    # bias-corrected rms as percentage of true value
                    corrected_rms_percent = 100 * corrected_rms / true_val
                    
                    print(f"{config_name:<30} {years:<8.1f} {minuit_stat_precision:>18.2f} "
                          f"{bias:>15.1f} {corrected_rms_percent:>20.2f}")
                else:
                    print(f"{config_name:<30} {years:<8.1f} {'n/a':>18} {'n/a':>15} {'n/a':>20}")
            else:
                print(f"{config_name:<30} {years:<8.1f} {'n/a':>18} {'n/a':>15} {'n/a':>20}")
    
    print("="*140)
    print(f"minuit stat = average minuit error as % of fitted value")
    print(f"bias-corr rms = bias-corrected rms on {signal_channel} as % of true value")
    print()

if __name__ == "__main__":
    
    print("="*80)
    print("plotting combined results")
    print("="*80)
    print(f"results directory: {cfg.RESULTS_DIR}")
    print(f"filter scenario: {FILTER_FIT_SCENARIO}")
    print(f"filter dimension: {FILTER_FIT_DIMENSION}")
    print()
    
    # load all results
    all_results = load_all_results(cfg.RESULTS_DIR, 
                                    filter_scenario=FILTER_FIT_SCENARIO,
                                    filter_dimension=FILTER_FIT_DIMENSION)
    
    if len(all_results) == 0:
        print("\nno results found matching filters!")
        exit(1)
    
    print(f"\nloaded {len(all_results)} configurations")
    
    # create summary table
    create_comparison_table(all_results)
    
    # determine detector types present
    has_water = any(k.startswith('water') for k in all_results.keys())
    has_wbls = any(k.startswith('1wbls') for k in all_results.keys())
    
    scenario_str = FILTER_FIT_SCENARIO if FILTER_FIT_SCENARIO else "all"
    dimension_str = FILTER_FIT_DIMENSION if FILTER_FIT_DIMENSION else "all"
    
    # create separate plots for each detector type
    if has_water:
        output_path = cfg.HISTS_DIR / f"precision_curves_{scenario_str}_{dimension_str}_water.png"
        plot_precision_curves(all_results, output_path, detector_filter='water')
    
    if has_wbls:
        output_path = cfg.HISTS_DIR / f"precision_curves_{scenario_str}_{dimension_str}_wbls.png"
        plot_precision_curves(all_results, output_path, detector_filter='1wbls')
    
    # also create combined plot if both detector types present
    if has_water and has_wbls:
        output_path = cfg.HISTS_DIR / f"precision_curves_{scenario_str}_{dimension_str}_combined.png"
        plot_precision_curves(all_results, output_path, detector_filter=None)
        print("\nℹ note: created separate plots for water and wbls, plus combined plot")
    
    print("\n" + "="*80)
    print("plotting complete")
    print("="*80)
