PARTITION=${PARTITION:-"VC2"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
STRIDE=${STRIDE:-"-1"}

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

declare -a tasks=( \
   'ALFRED' \
    'ActionLocalization' \
    'ActionPrediction' \
    'ActionSequence' \
   'CLEVR-Change' \
   'CharacterOrder' \
   'CounterfactualInference' \
   'DocVQA' \
   'EgocentricNavigation' \
   'GPR1200' \
   'IEdit' \
   'ImageNeedleInAHaystack' \
   'MMCoQA' \
    'MovingAttribute' \
   'MovingDirection' \
   'MultiModalQA' \
   'OCR-VQA' \
    'ObjectExistence' \
    'ObjectInteraction' \
   'ObjectShuffle' \
   'SceneTransition' \
   'SlideVQA' \
   'Spot-the-Diff' \
   'StateChange' \
   'TQA' \
   'TextNeedleInAHaystack' \
   'WebQA' \
   'WikiVQA' \
   'nuscenes' \
)

if [ "$STRIDE" = "-1" ]; then
    LOG_DIR=$CHECKPOINT/eval_milebench
else
    LOG_DIR=$CHECKPOINT/eval_milebench_${STRIDE}
fi

mkdir -p $LOG_DIR

model_name="internvl"

for ((j=0; j<${#tasks[@]}; j++)); do

    task="${tasks[j]}"

    
    echo "$(date) ${model_name}_${task}"

    srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=$((GPUS / GPUS_PER_TASK)) \
            --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
            --quotatype=${QUOTA_TYPE} \
            --job-name="${STRIDE}_${task}" \
            -o "${LOG_DIR}/${task}.log" \
            -e "${LOG_DIR}/${task}.log" \
            --async \
            python -u eval/milebench/eval_milebench.py \
            --checkpoint $CHECKPOINT \
            --output_dir $LOG_DIR \
            --dataset_name $task \
            --num-gpus-per-rank ${GPUS_PER_TASK} 
    sleep 0.2

done
