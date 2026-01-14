# Instructions

1. Edit `config.py` to set to preprocessing directory paths and fit variables 
2. Edit `preprocess_data.py` to set the input root files you want to use
- All root files are at `/nfs/disk1/users/bharris/eos/sim/outputs/sns/`
3. Run `python3 preprocess_data.py` 
- Skips files that already exist
- All already preprocessed npz files are at `/nfs/disk1/users/bharris/eos/sim/preprocessed_data_merged/`
4. Run the likelihood fit 
- 4.1 Fit to a single config with `python3 fit_single_config.py <detector> <shielding> <neutrons_scaling> <fit_scenario> <fit_dimension>`. Does save energy direction histograms to `hists` folder
    - For example `python3 fit_single_config.py <water/1wbls> <0ft/1ft/3ft> <0/1/100> <oxygen/galllium> <1D/2D>`
- 4.2 Fit to all configs defined in `config.py` with `python3 fit_all_configs.py`. Does save energy direction histograms to `hists` folder
- 4.3 Fit to all configs defined in `config.py` in parallel with batch jobs defined in `config.py`. Useful when number of toy datasets is large. Does not save energy direction histograms to `hists` folder
5. Plot results with `python3 plot_results.py`
- This only creats fit bias histograms and final cross section sensitivity plots
- Prints out cross section sensitivities for all configs as a function of exposure time
6. Use 'final_plots.ipynb` to make pretty plots with print out

### `config.py`

Sets variables for... 

#### Processing data (should only do once)
- Path to the root files of EOS detector simulations (relative to this directory)
- Total charge to reconstructed energy conversion factors in water and wbls
- Where to store the preprocessed npz files that contain reconstructed energy and direction of every event in the root files (relative to this directory)

#### Running the likelihood fit (uses npz files instead of root files)
- Energy range of the fit
- Number of energy and direction bins 
- SNS Neutrino flux uncertainties
- Event rates for 1 SNS year
- Neutron scaling factors
- SNS beam power
- Which detector configurations to run
    - water/wbls
    - amount of neutron sheilding 0/1/3ft
    - neutron rate scaling 0/1/10/100x
- SNS exposure times to fit to
- Number of toy datasets to fit to for each exposure time using asimov pdf
- Fraction of all data used to make asimov pdf (does not overlap with event pool toys are sampled from)
- How to smooth the asimov pdfs (leave off - high stats make it smooth)

#### Plotting results
- Channel names
- Channel mapping
- Channel colors 

