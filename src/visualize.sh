

cd $HOME/MBSNetwork/MBSNetwork
set -a
source .env.hpc
set +a
export PYTHONPATH=$PWD/src



poetry run python -m network.visualization
