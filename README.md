sns-xscn-analysis/
├── config.py             # Centralized configuration (paths, constants, settings)
├── analysis_utils.py     # Shared analysis functions (loading, filtering, fitting)
├── plotting_utils.py     # Shared plotting functions
├── fitter.py             # Sequential analysis of all configurations
├── test_single_config.py # Single configuration analysis (for parallel execution)
├── submit_jobs.sh        # SLURM job submission script
├── plot_results.py       # Plot combined results from parallel runs
├── preprocess_data.py    # Convert ROOT files to compressed NPZ format
├── results/              # Output directory for fit results (auto-created)
├── hists/                # Output directory for histogram plots (auto-created)
└── log/                  # SLURM log files (auto-created)
