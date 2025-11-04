#!/bin/bash
#
# Submit all sensitivity analysis jobs to SLURM
# This script loops through all configurations and submits one job per config
#

# Base directory
BASE_DIR="/nfs/disk1/users/bharris/eos/sim/eos-sns-analysis/oxygen_analysis/new"
LOG_DIR="${BASE_DIR}/log"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Define all configurations
# Format: detector shielding beam_power
CONFIGS=(
    "water 0ft 10"
    "water 0ft 100"
    "water 1ft 10"
    "water 1ft 100"
    "water 3ft 10"
    "water 3ft 100"
    "1wbls 0ft 10"
    "1wbls 0ft 100"
    "1wbls 1ft 10"
    "1wbls 1ft 100"
    "1wbls 3ft 10"
    "1wbls 3ft 100"
)

# Fit scenarios to run (can be "oxygen", "gallium", or both)
FIT_SCENARIOS=("oxygen")

# Fit dimensions to run (can be "1D", "2D", or both)
FIT_DIMENSIONS=("1D" "2D")

# Loop through all combinations
echo "Submitting jobs..."
echo "=================="

for config in "${CONFIGS[@]}"; do
    # Parse config
    read -r detector shielding beam_power <<< "$config"
    
    for fit_scenario in "${FIT_SCENARIOS[@]}"; do
        for fit_dimension in "${FIT_DIMENSIONS[@]}"; do
            
            # Create job name
            job_name="${detector}_${shielding}_${beam_power}MW_${fit_scenario}_${fit_dimension}"
            log_file="${LOG_DIR}/${job_name}_%j.log"
            
            # Submit job
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2gb
#SBATCH --time=00:30:00
#SBATCH --output=${log_file}
#SBATCH --partition=ubuntu_short

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mypythonenv

# Change to working directory
cd ${BASE_DIR}

# Print some info
echo "Running on host: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Cores: \$SLURM_CPUS_PER_TASK"
echo "Conda environment: \$CONDA_DEFAULT_ENV"
echo ""
echo "Configuration:"
echo "  Detector: ${detector}"
echo "  Shielding: ${shielding}"
echo "  Beam Power: ${beam_power} MW"
echo "  Fit Scenario: ${fit_scenario}"
echo "  Fit Dimension: ${fit_dimension}"
echo ""

# Run the analysis
python test_single_config.py ${detector} ${shielding} ${beam_power} ${fit_scenario} ${fit_dimension}

EOF
            
            echo "Submitted: ${job_name}"
            
        done
    done
done

echo "=================="
echo "All jobs submitted!"
echo ""
echo "To check job status: squeue -u \$USER"
echo "To check logs: ls -lth ${LOG_DIR}/"
