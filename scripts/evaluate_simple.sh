set -x

STRIDE=${STRIDE:-"-1"}
CHECKPOINT=${1}

ARGS=("$@")


declare -a tasks=( \
    'scienceqa' \
    'mmmu-val' \
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
    CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 6 "${ARGS[@]:1}" > "$CHECKPOINT/eval_origin_$STRIDE/${task}.log"
done
