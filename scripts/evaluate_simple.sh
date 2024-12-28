set -x

STRIDE=${STRIDE:-"-1"}
GPUS=${GPUS:-1"-1"}
CHECKPOINT=${1}

ARGS=("$@")

declare -a tasks=( \
    'mmmu-val-cot' \
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
    GPUS=${GPUS} bash scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 6 "${ARGS[@]:1}" 
    # > "$CHECKPOINT/eval_origin_$STRIDE/${task}.log"
done