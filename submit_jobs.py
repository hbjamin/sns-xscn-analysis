#!/bin/bash

# get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
LOG_DIR="${BASE_DIR}/log"

# create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# define all configurations
# format: detector shielding neutrons_per_mw
CONFIGS=(
    "water 0ft 10"
    "water 0ft 100"
    "water 1ft 10"
    "water 1ft 100"
    "water 3ft 10"
    "water 3ft 100"
)

# fit scenarios to run
FIT_SCENARIOS=("oxygen")

# fit dimensions to run
FIT_DIMENSIONS=("1D" "2D")

# loop through all combinations
echo "submitting jobs..."
echo "=================="
echo "base directory: ${BASE_DIR}"
echo ""

for config in "${CONFIGS[@]}"; do
    # parse config
    read -r detector shielding neutrons_per_mw <<< "$config"
    
    for fit_scenario in "${FIT_SCENARIOS[@]}"; do
        for fit_dimension in "${FIT_DIMENSIONS[@]}"; do
            
            # create job name
            job_name="${detector}_${shielding}_${neutrons_per_mw}npmw_${fit_scenario}_${fit_dimension}"
            log_file="${LOG_DIR}/${job_name}_%j.log"
            
            # submit job
            sbatch <<INNEREOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2gb
#SBATCH --time=00:30:00
#SBATCH --output=${log_file}
#SBATCH --partition=ubuntu_short

# load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mypythonenv

# change to working directory
cd ${BASE_DIR}

# print some info
echo "running on host: \$(hostname)"
echo "job id: \$SLURM_JOB_ID"
echo "cores: \$SLURM_CPUS_PER_TASK"
echo "conda environment: \$CONDA_DEFAULT_ENV"
echo "working directory: \$(pwd)"
echo ""
echo "configuration:"
echo "  detector: ${detector}"
echo "  shielding: ${shielding}"
echo "  neutrons per mw: ${neutrons_per_mw}"
echo "  fit scenario: ${fit_scenario}"
echo "  fit dimension: ${fit_dimension}"
echo ""

# run the analysis
python test_single_config.py ${detector} ${shielding} ${neutrons_per_mw} ${fit_scenario} ${fit_dimension}

INNEREOF
            
            echo "submitted: ${job_name}"
            
        done
    done
done

echo "=================="
echo "all jobs submitted!"
echo ""
echo "to check job status: squeue -u \$USER"
echo "to check logs: ls -lth ${LOG_DIR}/"
echo ""
echo "after all jobs complete, run:"
echo "  python plot_results.py"
