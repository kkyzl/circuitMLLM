[user@cbi-lgn01 ]$ vi pytorch_synthetic_benchmark.sh
#!/bin/bash
#SBATCH --job-name=singularity    ## job name
#SBATCH --nodes=2                ## 索取 2 節點
#SBATCH --ntasks-per-node=4      ## 每個節點運行 4 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:4
#SBATCH --account=GOV113XXX   ## PROJECT_ID
#SBATCH --partition=normal        ##
#SBATCH -o %j_mine.out                # Path to the standard output file
#SBATCH -e %j_mine.err                # Path to the standard error ouput file

module purge
ml singularity

export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=1
export HOROVOD_CPU_OPERATIONS=CCL
export NCCL_DEBUG=WARN


SIF=/work/hpc_sys/sifs/pytorch_22.09-py3_horovod.sif
SINGULARITY="singularity run --nv --no-home -B .:/data $SIF"
HOROVOD="python /data/pytorch_synthetic_benchmark.py --batch-size 256"

export HOROVOD_CPU_OPERATIONS=CCL
export NCCL_DEBUG=WARN
srun --mpi=pmi2 "${SINGULARITY[@]}" "${HOROVOD[@]}"