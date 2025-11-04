#!/usr/bin/env python
"""
Plot Combined Results

Reads all saved result files and creates combined precision curves.
Can filter by fit_scenario and fit_dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import glob
import os

hep.style.use("ROOT")

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = "/nfs/disk1/users/bharris/eos/sim/eos-sns-analysis/oxygen_analysis/new/results"
OUTPUT_DIR = "/nfs/disk1/users/bharris/eos/sim/eos-sns-analysis/oxygen_analysis/new"

# Filter settings (set to None to include all)
FILTER_FIT_SCENARIO = "oxygen"  # "oxygen", "gallium", or None
FILTER_FIT_DIMENSION = "2D"     # "1D", "2D", or None

# Plot settings
COLORS = plt.cm.tab10.colors
MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', 'p']

# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_all_results(results_dir, filter_scenario=None, filter_dimension=None):
    """Load all result pickle files from directory."""
    
    result_files = glob.glob(os.path.join(results_dir, "results_*.pkl"))
    
    if len(result_files) == 0:
        print(f"ERROR: No result files found in {results_dir}")
        return {}
    
    print(f"Found {len(result_files)} result files")
    
    all_results = {}
    
    for result_file in result_files:
        try:
            with open(result_file, 'rb') as f:
                data = pickle.load(f)
            
            config = data['config']
            
            # Apply filters
            if filter_scenario is not None and config['fit_scenario'] != filter_scenario:
                continue
            if filter_dimension is not None and config['fit_dimension'] != filter_dimension:
                continue
            
            # Create config key
            config_key = f"{config['detector']}_{config['shielding']}_{config['beam_power']}MW"
            
            # Store data
            all_results[config_key] = {
                'config': config,
                'signal_channel': data['signal_channel'],
                'exposure_times': data['exposure_times'],
                'results': data['results']
            }
            
            print(f"  Loaded: {config_key} ({config['fit_scenario']}, {config['fit_dimension']})")
            
        except Exception as e:
            print(f"  ERROR loading {result_file}: {e}")
            continue
    
    return all_results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_precision_curves(all_results, output_path):
    """Create precision curve plot from all results - shows BOTH Minuit stat and bias-corrected RMS."""
    
    if len(all_results) == 0:
        print("ERROR: No results to plot!")
        return
    
    # Get signal channel and exposure times from first result
    first_result = list(all_results.values())[0]
    signal_channel = first_result['signal_channel']
    exposure_times = first_result['exposure_times']
    fit_scenario = first_result['config']['fit_scenario']
    fit_dimension = first_result['config']['fit_dimension']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot each config - BOTH metrics
    for idx, (config_name, result_data) in enumerate(sorted(all_results.items())):
        results = result_data['results']
        
        minuit_precisions = []
        bias_corrected_precisions = []
        
        for years in exposure_times:
            if years in results and len(results[years]) > 0:
                # Get errors and fitted values from valid fits
                errors = [r['error'] for r in results[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in results[years] if r['valid']]
                
                if len(errors) > 0 and len(fitted_vals) > 0:
                    # Get true value
                    true_val = results[years][0]['true_value']
                    
                    # Calculate Minuit statistical precision
                    avg_error = np.mean(errors)
                    avg_fitted = np.mean(fitted_vals)
                    minuit_precision = 100 * avg_error / avg_fitted
                    minuit_precisions.append(minuit_precision)
                    
                    # Calculate bias-corrected RMS
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
        
        # Plot both curves with same color but different styles
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        
        # Bias-corrected RMS (solid line, filled marker)
        ax.plot(exposure_times, bias_corrected_precisions, 
                marker=marker, linestyle='-', 
                label=f"{config_name} (Bias-Corr RMS)", 
                linewidth=2.5, markersize=9, color=color, alpha=0.9)
        
        # Minuit stat (dashed line, hollow marker)
        ax.plot(exposure_times, minuit_precisions, 
                marker=marker, linestyle='--', 
                label=f"{config_name} (Minuit Stat)", 
                linewidth=2, markersize=8, color=color, alpha=0.6,
                fillstyle='none', markeredgewidth=2)
    
    # Formatting
    ax.set_xlabel("Years of Exposure", fontsize=16)
    ax.set_ylabel(f"Statistical Precision on {signal_channel} (%)", fontsize=16)
    ax.set_title(f"Statistical Precision vs. Exposure Time\n{fit_scenario.capitalize()} Sensitivity ({fit_dimension} Fit)\nSolid = Bias-Corrected RMS, Dashed = Minuit Statistical Error", 
                 fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.tick_params(labelsize=14)
    
    # Set reasonable axis limits
    ax.set_xlim(min(exposure_times) - 0.1, max(exposure_times) + 0.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved precision curves: {output_path}")
    plt.close()


def create_comparison_table(all_results):
    """Create a summary table of all results."""
    
    if len(all_results) == 0:
        print("ERROR: No results to summarize!")
        return
    
    print("\n" + "="*140)
    print("SUMMARY TABLE")
    print("="*140)
    
    # Get exposure times from first result
    first_result = list(all_results.values())[0]
    exposure_times = first_result['exposure_times']
    signal_channel = first_result['signal_channel']
    
    # Print header
    print(f"{'Config':<30} {'Years':<8} {'Minuit Stat (%)':>18} {'Avg Bias':>15} {'Bias-Corr RMS (%)':>20}")
    print("-"*140)
    
    # Print each config
    for config_name, result_data in sorted(all_results.items()):
        results = result_data['results']
        
        for years in exposure_times:
            if years in results and len(results[years]) > 0:
                errors = [r['error'] for r in results[years] if r['valid']]
                fitted_vals = [r['fitted'] for r in results[years] if r['valid']]
                
                if len(errors) > 0 and len(fitted_vals) > 0:
                    # Get metrics
                    avg_error = np.mean(errors)
                    avg_fitted = np.mean(fitted_vals)
                    true_val = results[years][0]['true_value']
                    
                    # Calculate bias
                    bias = avg_fitted - true_val
                    
                    # Calculate bias-corrected RMS around truth
                    corrected_rms = np.sqrt(np.mean([(v - bias - true_val)**2 for v in fitted_vals]))
                    
                    # Minuit statistical precision
                    minuit_stat_precision = 100 * avg_error / avg_fitted
                    
                    # Bias-corrected RMS as percentage of true value
                    corrected_rms_percent = 100 * corrected_rms / true_val
                    
                    print(f"{config_name:<30} {years:<8.1f} {minuit_stat_precision:>18.2f} {bias:>15.1f} {corrected_rms_percent:>20.2f}")
                else:
                    print(f"{config_name:<30} {years:<8.1f} {'N/A':>18} {'N/A':>15} {'N/A':>20}")
            else:
                print(f"{config_name:<30} {years:<8.1f} {'N/A':>18} {'N/A':>15} {'N/A':>20}")
    
    print("="*140)
    print(f"Minuit Stat = Average Minuit error as % of fitted value")
    print(f"Bias-Corr RMS = Bias-corrected RMS on {signal_channel} as % of true value (plotted metric)")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("PLOTTING COMBINED RESULTS")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Filter scenario: {FILTER_FIT_SCENARIO}")
    print(f"Filter dimension: {FILTER_FIT_DIMENSION}")
    print()
    
    # Load all results
    all_results = load_all_results(RESULTS_DIR, 
                                    filter_scenario=FILTER_FIT_SCENARIO,
                                    filter_dimension=FILTER_FIT_DIMENSION)
    
    if len(all_results) == 0:
        print("\nNo results found matching filters!")
        exit(1)
    
    print(f"\nLoaded {len(all_results)} configurations")
    
    # Create summary table
    create_comparison_table(all_results)
    
    # Create precision plot
    scenario_str = FILTER_FIT_SCENARIO if FILTER_FIT_SCENARIO else "all"
    dimension_str = FILTER_FIT_DIMENSION if FILTER_FIT_DIMENSION else "all"
    output_path = os.path.join(OUTPUT_DIR, f"precision_curves_{scenario_str}_{dimension_str}.png")
    
    plot_precision_curves(all_results, output_path)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
