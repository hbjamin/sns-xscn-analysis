#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import glob

import config as cfg
import plotting_utils as pu

hep.style.use("ROOT")

# filter settings (set to none to process all)
FILTER_FIT_SCENARIO = "oxygen"  # oxygen/gallium/none
FILTER_FIT_DIMENSION = "2D"     # 1d/2d/none

def load_all_results(results_dir, filter_scenario=None, filter_dimension=None):
    """
    Load all result pickle files and return in standardized format.
    
    Returns
    -------
    results_for_plotting : dict
        Results in format: {config_key: {year: [fit_results]}}
    metadata : dict
        Metadata from results: signal_channel, exposure_times, fit_scenario, fit_dimension
    """
    result_files = sorted(results_dir.glob("results_*.pkl"))
    
    if len(result_files) == 0:
        print(f"error: no result files found in {results_dir}")
        return {}, {}
    
    print(f"found {len(result_files)} result files")
    
    results_for_plotting = {}
    metadata = {
        'signal_channel': None,
        'exposure_times': None,
        'fit_scenario': None,
        'fit_dimension': None
    }
    
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
            
            # Extract just the results dict (not the wrapper)
            # This matches the format used by fit_all_configs.py
            results_for_plotting[config_key] = data['results']
            
            # Store metadata from first file
            if metadata['signal_channel'] is None:
                metadata['signal_channel'] = data['signal_channel']
                metadata['exposure_times'] = data['exposure_times']
                metadata['fit_scenario'] = config['fit_scenario']
                metadata['fit_dimension'] = config['fit_dimension']
            
            print(f"  loaded: {config_key} ({config['fit_scenario']}, {config['fit_dimension']})")
            
        except Exception as e:
            print(f"  error loading {result_file}: {e}")
            continue
    
    return results_for_plotting, metadata

def create_comparison_table(all_results, metadata):
    """
    Print a comparison table of results across all configurations.
    
    Parameters
    ----------
    all_results : dict
        Results dict: {config_key: {year: [fit_results]}}
    metadata : dict
        Contains signal_channel and exposure_times
    """
    if len(all_results) == 0:
        print("error: no results to summarize!")
        return
    
    signal_channel = metadata['signal_channel']
    exposure_times = metadata['exposure_times']
    
    print("\n" + "="*140)
    print("summary table")
    print("="*140)
    
    # print header
    print(f"{'config':<30} {'years':<8} {'minuit stat (%)':>18} {'avg bias':>15} {'bias-corr rms (%)':>20}")
    print("-"*140)
    
    # print each config
    for config_name, result_data in sorted(all_results.items()):
        
        for years in exposure_times:
            if years in result_data and len(result_data[years]) > 0:
                errors = [r['error'] for r in result_data[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in result_data[years] if r['valid']]
                
                if len(errors) > 0 and len(fitted_vals) > 0:
                    # get metrics
                    avg_error = np.mean(errors)
                    avg_fitted = np.mean(fitted_vals)
                    true_val = result_data[years][0]['true_value']
                    
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
    
    # load all results using standardized format
    results_for_plotting, metadata = load_all_results(
        cfg.RESULTS_DIR, 
        filter_scenario=FILTER_FIT_SCENARIO,
        filter_dimension=FILTER_FIT_DIMENSION
    )
    
    if len(results_for_plotting) == 0:
        print("\nno results found matching filters!")
        exit(1)
    
    print(f"\nloaded {len(results_for_plotting)} configurations")
    
    # create summary table
    create_comparison_table(results_for_plotting, metadata)
    
    # Use the SAME plotting functions as fit_all_configs.py
    # This is the key change - now both paths use identical plotting code
    print("\n" + "="*80)
    print("generating plots using shared plotting functions")
    print("="*80)
    
    output_path = cfg.HISTS_DIR / f'precision_curves_{metadata["fit_scenario"]}_{metadata["fit_dimension"]}.png'
    pu.plot_precision_curves(
        results_for_plotting, 
        metadata['exposure_times'], 
        metadata['signal_channel'],  # Pass actual signal_channel, not scenario
        metadata['fit_dimension'], 
        output_path
    )

    output_path = cfg.HISTS_DIR / f'bias_curves_{metadata["fit_scenario"]}_{metadata["fit_dimension"]}.png'
    pu.plot_bias_curves(
        results_for_plotting, 
        metadata['exposure_times'], 
        metadata['signal_channel'],  # Pass actual signal_channel, not scenario
        metadata['fit_dimension'], 
        output_path
    )
    
    print("\n" + "="*80)
    print("plotting complete")
    print("=" * 80)
