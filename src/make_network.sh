#!/bin/bash
#SBATCH --job-name=create_network
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


poetry run python -m network.network
poetry run python -m network.cooccurrence
poetry run python -m network.modularity
