set -x

PARTITION=${PARTITION:-"VC2"}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
STRIDE=${STRIDE:-"-1"}
CHECKPOINT=${1}

ARGS=("$@")


declare -a tasks=( \
    'vqa-chartqa-test' \
    'vqa-docvqa-val' \
    'vqa-ai2d-test' \
    'vqa-infovqa-val' \
    'scienceqa' \
    'pope' \
    'mmmu-val' \
    'mmbench-test-en' \
    'seed' \
)

if [ "$STRIDE" = "-1" ]; then
    mkdir -p "$CHECKPOINT/eval_origin"
else
    mkdir -p "$CHECKPOINT/eval_origin_${STRIDE}"
fi

for ((j=0; j<${#tasks[@]}; j++)); do
    model_path=$CHECKPOINT
    task=${tasks[j]}

    model_name="$(basename ${model_path})"
   
    echo "$(date) ${model_name}_${task}"

    if [ "${task}" == "vqa-chartqa-test" ]; then
        srun \
            -p ${PARTITION} \
            --gres=gpu:8 \
            --ntasks=1 \
            --quotatype=${QUOTA_TYPE} \
            --ntasks-per-node=1 \
            -o "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            -e "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            --async \
        sh scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 12 "${ARGS[@]:1}"
    elif [ "${task}" == "vqa-infovqa-val" ]; then
        srun \
            -p ${PARTITION} \
            --gres=gpu:8 \
            --ntasks=1 \
            --quotatype=${QUOTA_TYPE} \
            --ntasks-per-node=1 \
            -o "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            -e "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            --async \
        sh scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 24 "${ARGS[@]:1}"
    elif [ "${task}" == "vqa-docvqa-val" ]; then
        srun \
            -p ${PARTITION} \
            --gres=gpu:8 \
            --ntasks=1 \
            --quotatype=${QUOTA_TYPE} \
            --ntasks-per-node=1 \
            -o "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            -e "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            --async \
        sh scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 18 "${ARGS[@]:1}"
    else
        srun \
            -p ${PARTITION} \
            --gres=gpu:8 \
            --ntasks=1 \
            --quotatype=${QUOTA_TYPE} \
            --ntasks-per-node=1 \
            -o "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            -e "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
            --async \
        sh scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 6 "${ARGS[@]:1}"
    fi
done
