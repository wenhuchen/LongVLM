PARTITION=${PARTITION:-"VC2"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-2}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

set -x


CHECKPOINT=${1}
task="arxiv_caption"
LOG_DIR=$CHECKPOINT/infer
mkdir -p $LOG_DIR

srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=$((GPUS / GPUS_PER_TASK)) \
            --jobid=3612349 \
            --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
            --quotatype=${QUOTA_TYPE} \
            --job-name=${task} \
            python -u eval/infer/infer.py \
            --checkpoint $CHECKPOINT \
            --outputs-dir $LOG_DIR \
            --task $task \
            --num-gpus-per-rank ${GPUS_PER_TASK} 
  2>&1 | tee -a "${LOG_DIR}/training_log.txt"