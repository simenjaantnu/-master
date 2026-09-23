#!/bin/bash
#SBATCH --job-name=calculatethreshold
#SBATCH --output=output_%j.txt

#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
set -euo pipefail


cd $HOME/MBSNetwork/MBSNetwork
set -a
source .env.hpc
set +a
export PYTHONPATH=$PWD/src


poetry run python -m network.threshold
