#!/bin/bash
#SBATCH --job-name=drugdisc
#SBATCH --output=output_%j.txt

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
set -euo pipefail


cd $HOME/MBSNetwork/MBSNetwork
set -a
source .env.hpc
set +a
export PYTHONPATH=$PWD/src


poetry run python -m analysis.dataset
poetry run python -m analysis.drugs
poetry run python -m analysis.preliminary
poetry run python -m analysis.sequence
