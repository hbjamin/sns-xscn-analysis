# How to use this repo

### File paths are hardcoded inside scripts

- First load the reconstructed energies and directions from the root files on nubar to .npz files. A conversion factor is used to go from event total charge to reconstructed energy 
```bash
python3 preprocess_data.py
```
- Then adjust the configuration in `fitter.py` and run. Stats printed out to terminal. Histograms of asimov pdf and fit datasets created for each fit scenario and exposure time. Limit the number of toy datasets used or else it will be too slow.
```bash
python3 fitter.py
```
- If you want more toy datasets, submit jobs to the cluster in parallel. This will submit one job for each fit scenario.
```bash
sbatch submit_jobs.py
```
- Then plot then results that compare minuit statistical precision with bias corrected rms 
```bash
python3 plot_results.py
```


