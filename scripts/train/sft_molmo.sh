export WANDB_PROJECT="SAGE"
export RUN_NAME=SAGE-MM-Molmo2-D-8B-SFT

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    sage/src/train/sft.py \
    --output_dir "outputs/${RUN_NAME}" \
    --model_name_or_path "allenai/Molmo2-D-8B" \
    --dataset_name "allenai/SAGE-MM-SFT-417K" \
    --deepspeed scripts/deepspeed/zero2_offload.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --bf16 True \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 5000 \
    --max_grad_norm 5 \
    --save_only_model true
