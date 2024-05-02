MAX_STEPS=10000
LR=1e-5
BS=32
PROMPT_DROPOUT=0.05
OUTPUT_DIR="controlnet/minisd_${BS}_${LR}_${MAX_STEPS}_dropout${PROMPT_DROPOUT}"
MODEL_NAME="lambdalabs/miniSD-diffusers"

accelerate launch --multi_gpu train_controlnet.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --resolution=256 \
    --learning_rate=${LR} \
    --max_train_steps=${MAX_STEPS} \
    --max_train_samples=80000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="/data/laion400m-data/{00000..10200}.tar" \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.jpeg" \
    --validation_prompt "a dog sitting on the grass" "home office" \
    --validation_steps=100 \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --train_batch_size=${BS} \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --mixed_precision="fp16" \
    --tracker_project_name="controlnet" \
    --report_to=wandb \
    --proportion_empty_prompts ${PROMPT_DROPOUT}