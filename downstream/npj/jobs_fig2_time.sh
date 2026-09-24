#!/bin/bash
# Time-to-event registry model, fig2_time_to_rx.py: four chains per censoring rule, one
# chain per task, then summarise each rule once its four chains are on disk:
#   sbatch downstream/npj/jobs_fig2_time.sh
#   python3 downstream/fig2_time_to_rx.py --censor admin --summarise 1
#   python3 downstream/fig2_time_to_rx.py --censor last_record --summarise 1
#SBATCH --job-name=aud_fig2_time
#SBATCH --account=r00209
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/fig2_time_%A_%a.log
cd /N/project/depot/hw56/aud/medace_aud
module load python/gpu/3.12.5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
rules=(admin last_record)
rule=${rules[$((SLURM_ARRAY_TASK_ID / 4))]}
chain=$((SLURM_ARRAY_TASK_ID % 4))
python3 -u downstream/fig2_time_to_rx.py --censor "$rule" --chain "$chain"
