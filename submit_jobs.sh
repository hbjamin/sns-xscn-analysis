#!/bin/bash
#SBATCH --job-name=submit_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500mb
##SBATCH --time=00:05:00
#SBATCH --output=submit_jobs_%j.log
#SBATCH --partition=ubuntu_short

# CRITICAL: Set the actual project directory
# Replace this with your actual path
PROJECT_DIR="/nfs/disk1/users/bharris/eos/analysis/sns-xscn-analysis"

# Change to project directory
cd "${PROJECT_DIR}" || { echo "Failed to cd to ${PROJECT_DIR}"; exit 1; }

BASE_DIR="${PROJECT_DIR}"
LOG_DIR="${BASE_DIR}/log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "Project directory: ${PROJECT_DIR}"
echo "Base directory: ${BASE_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Current working directory: $(pwd)"
echo ""

# Define all configurations
# format: detector shielding neutrons_per_mw
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

# Fit scenarios to run
FIT_SCENARIOS=("oxygen")

# Fit dimensions to run
FIT_DIMENSIONS=("2D")

# Loop through all combinations
echo "Submitting jobs..."
echo "=================="

for config in "${CONFIGS[@]}"; do
    # Parse config
    read -r detector shielding neutrons_per_mw <<< "$config"
    
    for fit_scenario in "${FIT_SCENARIOS[@]}"; do
        for fit_dimension in "${FIT_DIMENSIONS[@]}"; do
            
            # Create job name
            job_name="${detector}_${shielding}_${neutrons_per_mw}npmw_${fit_scenario}_${fit_dimension}"
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
cd ${BASE_DIR} || { echo "Failed to cd to ${BASE_DIR}"; exit 1; }

# Print some info
echo "Running on host: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Cores: \$SLURM_CPUS_PER_TASK"
echo "Conda environment: \$CONDA_DEFAULT_ENV"
echo "Working directory: \$(pwd)"
echo ""
echo "Configuration:"
echo "  Detector: ${detector}"
echo "  Shielding: ${shielding}"
echo "  Neutrons per MW: ${neutrons_per_mw}"
echo "  Fit scenario: ${fit_scenario}"
echo "  Fit dimension: ${fit_dimension}"
echo ""

# Run the analysis
python fit_single_config.py ${detector} ${shielding} ${neutrons_per_mw} ${fit_scenario} ${fit_dimension}

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
echo ""
echo "After all jobs complete, run:"
echo "  cd ${BASE_DIR}"
echo "  python plot_results.py"
