#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import gc

import config as cfg
import analysis_utils as au
import plotting_utils as pu

def run_scenario_analysis(channel_cache, shielding, neutrons_per_mw, detector_name,
                         fit_scenario, fit_dimension):
    print(f"\n{'='*80}")
    print(f"scenario: {detector_name}_{neutrons_per_mw}n/mw_{shielding}")
    print(f"{'='*80}")
    
    # load neutrons and add to data
    print(f"\nloading neutrons:")
    energy_direction_data = {}
    for ch in channel_cache:
        energy, direction, ntrig, nsim = channel_cache[ch]
        energy_direction_data[ch] = (energy, direction, ntrig, nsim)
    
    try:
        neutron_data = au.load_and_spectrum_weight_neutrons(detector_name, shielding)
        energy_direction_data['neutrons'] = neutron_data
    except FileNotFoundError as e:
        print(f"  skipped: {e}")
        return None, None

    # filter data to analysis range (now returns neutron_metadata for year scaling)
    filtered_data, filtered_rates, neutron_metadata = au.filter_data_to_analysis_range(
        energy_direction_data, cfg.EVENT_RATES_TOTAL
    )
    
    # Calculate neutron rate here (once for all exposure times)
    if neutron_metadata:
        neutron_rate = au.calculate_neutron_rate_per_year(neutron_metadata, neutrons_per_mw)
        filtered_rates['neutrons'] = neutron_rate
        print(f"\nneutron rate set to: {neutron_rate:.1f}/year")
    
    # CRITICAL: split data into asimov and toy pools FIRST (no overlap!)
    # This is done on RAW events, before any smoothing
    print(f"\nsplitting data: {cfg.ASIMOV_FRACTION:.0%} for asimov, "
          f"{1-cfg.ASIMOV_FRACTION:.0%} for toys")
    asimov_data, toy_data = au.split_data_for_asimov_and_toys(
        filtered_data, cfg.ASIMOV_FRACTION
    )
    
    # create asimov histograms from asimov pool only
    print(f"\ncreating asimov pdf ({cfg.FIT_DIMENSION}) from asimov pool:")
    binning = cfg.get_binning(cfg.FIT_DIMENSION)
    
    if cfg.FIT_DIMENSION == "1D":
        asimov_hist = {key: np.histogram(asimov_data[key][0], binning)[0] 
                      for key in asimov_data}
    else:
        asimov_hist = {key: np.histogram2d(asimov_data[key][0], asimov_data[key][1], binning)[0] 
                      for key in asimov_data}
    
    for key in asimov_hist:
        print(f"  {key}: {np.sum(asimov_hist[key]):.0f} events in asimov histogram")
    
    # NEW: Apply smoothing to asimov histograms AFTER histogram creation
    # This is the statistically correct approach - we smooth the histogram itself,
    # not the underlying events. The toy pool remains completely unsmoothed.
    if cfg.SMOOTH_ASIMOV['enabled'] and cfg.FIT_DIMENSION == "1D":
        print(f"\nsmoothing asimov histograms (method: {cfg.SMOOTH_ASIMOV['method']}):")
        bin_centers = 0.5 * (cfg.ENERGY_BINS[1:] + cfg.ENERGY_BINS[:-1])
        
        for key in asimov_hist:
            if key in cfg.SMOOTH_ASIMOV['channels']:
                original_sum = np.sum(asimov_hist[key])
                
                # Smooth the histogram directly
                asimov_hist[key] = au.smooth_asimov_histogram(
                    asimov_hist[key],
                    cfg.SMOOTH_ASIMOV['method'],
                    cfg.SMOOTH_ASIMOV['params'],
                    bin_centers
                )
                
                # Renormalize to preserve total event count
                smoothed_sum = np.sum(asimov_hist[key])
                if smoothed_sum > 0:
                    asimov_hist[key] = asimov_hist[key] * (original_sum / smoothed_sum)
                
                print(f"  {key}: smoothed (preserved {original_sum:.0f} events)")
    elif cfg.SMOOTH_ASIMOV['enabled'] and cfg.FIT_DIMENSION == "2D":
        print(f"\nwarning: smoothing not yet implemented for 2D histograms")
        print(f"  (asimov histograms will remain unsmoothed)")

    # determine channels to fit
    channels, signal_channel = cfg.get_channels_for_scenario(fit_scenario)
    
    # results storage
    all_results = {years: [] for years in cfg.EXPOSURE_TIMES}
    
    for years in cfg.EXPOSURE_TIMES:
        print(f"\n{'='*60}")
        print(f"exposure: {years} years")
        print(f"{'='*60}")
        
        # normalize and scale asimov (just for plotting)
        asimov_normalized, asimov_scaled = au.normalize_and_scale_asimov(
            asimov_hist, years, filtered_rates
        )
        
        # make interpolated pdfs (for fitting)
        norm_pdf_luts, pdfs, bin_centers = au.make_normalized_interpolated_pdf(
            asimov_normalized, cfg.FIT_DIMENSION 
        )
        
        # plot asimov projections (once per scenario)
        if years == cfg.EXPOSURE_TIMES[0]:
            output_path = (cfg.HISTS_DIR / 
                          f'asimov_projections_{detector_name}_{neutrons_per_mw}npmw_{shielding}_{cfg.FIT_DIMENSION}.png')
            pu.plot_asimov_projections(asimov_scaled, years, output_path, cfg.FIT_DIMENSION)

        # calculate total expected events
        total_events = sum(filtered_rates[ch] for ch in filtered_rates.keys())
        
        # process toys one by one for memory efficiency
        # sample from toy pool (separate from asimov data!)
        print(f"\ngenerating and fitting {cfg.N_TOYS} toy datasets...")
        fit_results = []
        last_toy_hist = None
        
        for toy_idx in range(cfg.N_TOYS):
            # generate one toy dataset from toy pool
            toy_datasets = au.make_toy_datasets_with_poisson_and_flux(
                toy_data, filtered_rates, years, ngroups=1,
                neutron_metadata=neutron_metadata, neutrons_per_mw=neutrons_per_mw
            )
            
            # make histogram for this toy
            if cfg.FIT_DIMENSION == "1D":
                toy_hist = {key: np.histogram(toy_datasets[key][0][0], binning)[0] 
                           for key in toy_data}
            else:
                toy_hist = {key: np.histogram2d(toy_datasets[key][0][0], 
                                                toy_datasets[key][0][1], binning)[0] 
                           for key in toy_data}
            
            # sum all channels for fitting
            fit_data = np.sum([toy_hist[ch] for ch in toy_data.keys()], axis=0)
            
            # print first toy as example
            if toy_idx == 0:
                print(f"\nexample toy 0:")
                for ch in toy_data.keys():
                    print(f"  {ch}: {np.sum(toy_hist[ch]):.0f} events")
            
            # fit this toy
            try:
                m = au.fit_with_extended_binned_nll(
                    fit_data, channels, norm_pdf_luts, years, total_events,
                    fit_scenario, cfg.FIT_DIMENSION, binning, filtered_rates
                )
                fit_results.append(m)
                
                if toy_idx == 0:
                    print(f"\nexample fit result:")
                    for ch in channels:
                        print(f"  {ch}: {m.values[ch]:.1f} ± {m.errors[ch]:.1f}")
            except Exception as e:
                print(f"  error: fit {toy_idx+1} failed: {e}")
                continue
            
            # keep only the last toy histogram for plotting
            if toy_idx == cfg.N_TOYS - 1:
                last_toy_hist = toy_hist.copy()
            
            # free memory
            del toy_datasets
            del toy_hist
            del fit_data
            
            # explicit garbage collection every 50 toys
            if (toy_idx + 1) % 50 == 0:
                gc.collect()
            
            # progress updates
            if (toy_idx + 1) % 100 == 0:
                print(f"  progress: {toy_idx + 1}/{cfg.N_TOYS} toys completed...")
        
        print(f"  finished fitting all {cfg.N_TOYS} toys!")
        
        # plot asimov + last toy overlaid
        if last_toy_hist is not None:
            output_path = (cfg.HISTS_DIR / 
                          f'asimov_toys_{detector_name}_{neutrons_per_mw}npmw_{shielding}_{years}yr_{cfg.FIT_DIMENSION}.png')
            pu.plot_asimov_and_fit_group_projections(
                asimov_scaled, [last_toy_hist], years, output_path, cfg.FIT_DIMENSION, n_toys_to_plot=1
            )
            del last_toy_hist
        
        # store results
        if len(fit_results) == 0:
            print(f"\n  error: all fits failed for {years} year exposure!")
            continue
        
        true_val = filtered_rates[cfg.CHANNEL_MAPPING[signal_channel]] * years
        
        for i, m in enumerate(fit_results):
            all_results[years].append({
                'fitted': m.values[signal_channel],
                'error': m.errors[signal_channel],
                'valid': m.valid,
                'true_value': true_val
            })
        
        # print summary
        signal_vals = [m.values[signal_channel] for m in fit_results if m.valid]
        signal_err = [m.errors[signal_channel] for m in fit_results if m.valid]
        
        if len(signal_vals) > 0:
            mean_val = np.mean(signal_vals)
            mean_err = np.mean(signal_err)
            bias = mean_val - true_val
            corrected_rms = np.sqrt(np.mean([(v - bias - true_val)**2 for v in signal_vals]))
            
            print(f"\nsummary of {cfg.N_TOYS} {signal_channel} toy fit results:")
            print(f"  mean fitted val : {mean_val:.1f}")
            print(f"  mean fitted err : {mean_err:.1f}")
            print(f"  mean fitted err / truth : {100*mean_err/true_val:.2f}%")
            print(f"  rms fitted val  : {np.std(signal_vals):.1f}")
            print(f"  true value      : {true_val:.1f}")
            print(f"  bias (mean - true): {bias:.1f}")
            print(f"  bias-corrected rms: {corrected_rms:.1f} ({100*corrected_rms/true_val:.2f}%)")
    
    return all_results, signal_channel

if __name__ == "__main__":
    
    # configuration
    
    print(f"{'='*80}")
    print("sns cross-section sensitivity analysis")
    print(f"{'='*80}")
    print(f"fit scenario: {cfg.FIT_SCENARIO}")
    print(f"fit dimension: {cfg.FIT_DIMENSION}")
    print(f"n_toys: {cfg.N_TOYS}")
    print(f"exposure times: {cfg.EXPOSURE_TIMES}")
    print(f"asimov fraction: {cfg.ASIMOV_FRACTION:.0%}")
    print(f"smoothing enabled: {cfg.SMOOTH_ASIMOV['enabled']}")
    if cfg.SMOOTH_ASIMOV['enabled']:
        print(f"  method: {cfg.SMOOTH_ASIMOV['method']}")
        print(f"  channels: {cfg.SMOOTH_ASIMOV['channels']}")
        print(f"  note: smoothing applied to asimov HISTOGRAMS after splitting")
    print(f"\noutput directories:")
    print(f"  results: {cfg.RESULTS_DIR}")
    print(f"  histograms: {cfg.HISTS_DIR}")
    
    # process all configurations
    all_results_by_config = {}
    
    for detector_name, shielding, neutrons_per_mw in cfg.CONFIGS:
        # load channels once per detector type
        print(f"\n{'='*80}")
        print(f"loading data: {detector_name}_{neutrons_per_mw}n/mw_{shielding}")
        print(f"{'='*80}")
        
        channel_cache = {}
        channels_to_load = cfg.get_channels_to_load(cfg.FIT_SCENARIO)
        
        for channel_name in channels_to_load:
            print(f"  {channel_name}:")
            try:
                data = au.load_preprocessed_channel(detector_name, channel_name)
                channel_cache[channel_name] = data
            except FileNotFoundError as e:
                print(f"    error: {e}")
                continue
        
        # run analysis
        results, signal_channel = run_scenario_analysis(
            channel_cache, shielding, neutrons_per_mw, detector_name,
            cfg.FIT_SCENARIO, cfg.FIT_DIMENSION
        )
        
        if results is not None:
            config_key = f"{detector_name}_{shielding}_{neutrons_per_mw}npmw"
            all_results_by_config[config_key] = results
    
    # plot precision curves
    if len(all_results_by_config) > 0:
        print("\n" + "="*80)
        print("plotting precision curves")
        print("="*80)
        
        output_path = cfg.HISTS_DIR / f'precision_curves_{cfg.FIT_SCENARIO}_{cfg.FIT_DIMENSION}.png'
        pu.plot_precision_curves(
            all_results_by_config, cfg.EXPOSURE_TIMES, signal_channel,
            cfg.FIT_SCENARIO, cfg.FIT_DIMENSION, output_path
        )

        output_path = cfg.HISTS_DIR / f'bias_curves_{cfg.FIT_SCENARIO}_{cfg.FIT_DIMENSION}.png'
        pu.plot_bias_curves(
            all_results_by_config, cfg.EXPOSURE_TIMES, signal_channel,
            cfg.FIT_SCENARIO, cfg.FIT_DIMENSION, output_path
        )
    
    print("\n" + "=" * 80)
    print("analysis complete")
    print("=" * 80)
