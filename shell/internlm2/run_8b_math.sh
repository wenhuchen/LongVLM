OUTPUT_DIR="internlm2vl_v2pe_math"
MODEL_PATH="pretrained/InternVL2-8B"
# MODEL_PATH="pretrained/InternVL2-2B"

META_PATH="shell/data/annotation_train_math.json"

torchrun --nproc_per_node=8 internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${MODEL_PATH} \
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
  --max_steps 10000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
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
  --log_freq 1000 \
  --strict_mode False \
  --rope_pos_id_version 'v2pe_rnd' \
  --replacement False \
  --allow_overflow False \
  --remove_unused_columns False \
  --loss_reduction "square" \
  --loss_reduction_all_gather True \
  --num_images_expected 32 \
  --max_buffer_size 10 \
  --max_packed_tokens 16000 \
  2>&1 | tee output_file.txt