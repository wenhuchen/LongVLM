PARTITION=${PARTITION:-"VC2"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
STRIDE=${STRIDE:-"-1"}
GROUP=${GROUP:-"32"}


set -x

CHECKPOINT=${1}

JOB_FOLDER=$(dirname "$CHECKPOINT")
files=(
    "$JOB_FOLDER/configuration_intern_vit.py"
    "$JOB_FOLDER/configuration_internlm2.py"
    "$JOB_FOLDER/configuration_internvl_chat.py"
    "$JOB_FOLDER/conversation.py"
    "$JOB_FOLDER/modeling_intern_vit.py"
    "$JOB_FOLDER/modeling_internlm2.py"
    "$JOB_FOLDER/modeling_internvl_chat.py"
    "$JOB_FOLDER/tokenization_internlm2_fast.py"
    "$JOB_FOLDER/tokenization_internlm2.py"
)
for file in "${files[@]}"; do
    dest_file="$CHECKPOINT/$(basename "$file")"
    if [ ! -f "$dest_file" ]; then
        cp "$file" "$CHECKPOINT"
    fi
done
ARGS=("$@")

declare -a docs=( \
    'deepform' \
    'docvqa' \
    'infovqa' \
    'kleistercharity' \
    'svqa' \
    'visualmrc' \
)


declare -a tasks=( \
    'chartqa' \
    'clevr' \
    'deepform' \
    'docvqa' \
    'dvqa' \
    'gqa' \
    'infovqa' \
    'kleistercharity' \
    'ocrvqa' \
    'okvqa' \
    'svqa' \
    'tabfact' \
    'textcaps' \
    'textvqa' \
    'visualmrc' \
    'vizwiz' \
    'wikitablequestions' \
)

if [ "$STRIDE" = "-1" ]; then
    LOG_DIR=$CHECKPOINT/eval_longvqa
else
    LOG_DIR=$CHECKPOINT/eval_longvqa_${STRIDE}
fi

mkdir -p $LOG_DIR

model_name="internvl"

for ((j=0; j<${#tasks[@]}; j++)); do

    TASK=${tasks[j]}
    if [ "$GROUP" = "32" ]; then
        FILE="dataset/val/long_vqa_${GROUP}k/val_long_vqa_${task}_16_32.jsonl"
    else
        from=$((GROUP-8))
        FILE="dataset/val/long_vqa_${GROUP}k/val_long_vqa_${task}_${from}_${GROUP}.jsonl"
    fi

    if [[ " ${docs[@]} " =~ " $TASK " ]]; then
        ROOT="dataset/image/long_vqa/image/${TASK}/val"
    else
        ROOT="dataset/image/long_vqa/paste/${TASK}/val"
    fi

    TASK_FULL=longvqa_${TASK}_${GROUP}

    echo "$(date) ${model_name}_${TASK_FULL}"

    srun -p ${PARTITION} \
            --job-name=${STRIDE}_${TASK_FULL} \
            --gres=gpu:${GPUS_PER_NODE} \
            --async \
            --ntasks=$((GPUS / GPUS_PER_TASK)) \
            --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
            --quotatype=${QUOTA_TYPE} \
            -o "${LOG_DIR}/${TASK_FULL}.log" \
            -e "${LOG_DIR}/${TASK_FULL}.log" \
            python -u eval/longvqa/eval_longvqa.py \
            --checkpoint $CHECKPOINT \
            --task $TASK \
            --file $FILE \
            --root $ROOT \
            --outputs-dir $LOG_DIR/${TASK_FULL} \
            --num-gpus-per-rank ${GPUS_PER_TASK} "${ARGS[@]:1}"

    sleep 0.2

done
