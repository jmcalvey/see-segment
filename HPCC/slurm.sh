%%writefile submit_tpot_array.sb
#!/bin/bash --login
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail # Force quit if there is an error.

module purge
module load Miniforge3
conda activate ./envs

SEARCH_SEED=$((802 + SLURM_ARRAY_TASK_ID))
OUTPUT_DIR="results/seed-${SEARCH_SEED}"

python run_tpot_array.py \
  --search-seed "${SEARCH_SEED}" \
  --split-seed 802 \
  --max-time-mins 5 \
  --output-dir "${OUTPUT_DIR}"