set -x

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

SCRIPT_NAME=$(basename "$0" .sh)
OUTPUT_DIR="work_dirs/${SCRIPT_NAME}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
files=(
  "internvl/conversation.py"
  "internvl/model/internlm2/configuration_internlm2.py"
  "internvl/model/internlm2/modeling_internlm2.py"
  "internvl/model/internlm2/tokenization_internlm2_fast.py"
  "internvl/model/internlm2/tokenization_internlm2.py"
  "internvl/model/internvl_chat/configuration_intern_vit.py"
  "internvl/model/internvl_chat/configuration_internvl_chat.py"
  "internvl/model/internvl_chat/modeling_intern_vit.py"
  "internvl/model/internvl_chat/modeling_internvl_chat.py"
)


for file in "${files[@]}"; do
  cp "$file" "$OUTPUT_DIR"
done


MODEL_PATH="pretrained/InternVL2-2B"
META_PATH="shell/data/annotation_train_32k.json"


srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --job-name=32k \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u internvl/train/internvl_chat_finetune.py \
  --model_name_or_path $MODEL_PATH \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${META_PATH} \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 3 \
  --bf16 True \
  --num_train_epochs 1 \
  --max_steps 20000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2500 \
  --save_total_limit 5 \
  --learning_rate 5e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 34000 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --dynamic_max_patch False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard" \
  --use_packed_ds True \
  --num_images_expected 135 \
  --max_packed_tokens 34000 \
  --max_buffer_size 20 \
  --log_freq 1000 \
  --strict_mode False \
  --rope_pos_id_version 'v2pe_rnd' \
  --replacement False \
  --chunk_num 8 \
  --attn_type ring \
  --allow_overflow False \
  --remove_unused_columns False \
  --loss_reduction "square" \
  --loss_reduction_all_gather True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"