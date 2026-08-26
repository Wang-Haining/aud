#!/bin/bash
#SBATCH --job-name=aud_bayes_all
#SBATCH --account=r00209
#SBATCH --partition=general
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/bayes_all_%A_%a.log
cd /N/project/depot/hw56/aud/medace_aud
case $SLURM_ARRAY_TASK_ID in
  0) python3 -u downstream/fig6_bayes_strata.py ;;
  1) python3 -u downstream/fig6_bayes_strata.py --joint 1 ;;
  2) python3 -u downstream/fig6_bayes_strata.py --ever-rx 1 ;;
  3) python3 -u downstream/fig6_bayes_strata.py --variant acamp_disulf ;;
  4) python3 -u downstream/fig6_bayes_strata.py --variant theta_pooled ;;
  5) python3 -u downstream/fig6_bayes_strata.py --variant narrow_signal ;;
  6) python3 -u downstream/fig6_bayes_strata.py --collapse 1 ;;
esac
