#!/bin/bash

LLM_VERSION=Qwen2-VL-7B-Instruct
LLM_PATH="Qwen/Qwen2-VL-7B-Instruct"
SFT_TASK="stage2"
SAVE_DIR=results/aguvis/
IMAGE_FOLDER=data/aguvis/images/

SFT_DATA_YAML=data/${SFT_TASK}.yaml
SFT_RUN_NAME="${LLM_VERSION}-sft-${SFT_TASK}"
echo "SFT_RUN_NAME: ${SFT_RUN_NAME}"

# Define WORLD_SIZE (number of nodes)
WORLD_SIZE=1  # Adjust as necessary for multi-node setup
RANK=0  # Adjust for distributed training, typically for the first node
MASTER_ADDR=127.0.0.1
MASTER_PORT=29600

DISTRIBUTED_ARGS="\
    --nproc_per_node 8 \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
export ACCELERATE_CPU_AFFINITY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=c23bde849c64215088012dbabff2e9ef57479e23

LLM_PATH=${SAVE_DIR}/checkpoints/${LLM_VERSION}-sft-stage1
printenv
echo $LLM_PATH

torchrun $DISTRIBUTED_ARGS train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path ${SFT_DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --model_name_or_path $LLM_PATH \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_DIR}/checkpoints/${SFT_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --freeze_visual_encoder True \
    --resume_from_checkpoint ${SAVE_DIR}/checkpoints/${SFT_RUN_NAME} \
    --report_to "tensorboard" \
    --run_name "${SFT_RUN_NAME}" \
    > "${SAVE_DIR}/checkpoints/${SFT_RUN_NAME}.log" 2>&1