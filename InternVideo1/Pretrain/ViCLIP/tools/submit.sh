#!/usr/bin/env sh

mode=$1 # slurm or local
nnodes=$2
ngpus=$3
cmd=${@:4}  # the command to run. i.e. tasks/pretrain.py ...

if [[ "$mode" == "slurm" ]]; then # slurm
	master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
	all_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
	echo "All nodes used: ${all_nodes}"
	echo "Master node ${master_node}"

	head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address | awk '{print $1}')
	# head_node_ip=$master_node
	rdzv_endpoint="${head_node_ip}:${MASTER_PORT:-40000}"
	bin="srun"

else # local
	rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT:-40000}"
	bin=""
fi

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

#run command
$bin torchrun --nnodes=$nnodes \
	--nproc_per_node=$ngpus \
	--rdzv_backend=c10d \
	--rdzv_endpoint=${rdzv_endpoint} \
    $cmd

echo "Finish at dir: ${PWD}"
############### ======> Your training scripts [END]
