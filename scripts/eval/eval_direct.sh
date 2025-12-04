#!/bin/bash

export MODEL="Qwen/Qwen3-VL-8B-Instruct"
export BENCHMARK="sage_bench" # minerva_bench
export NUM_GPUS=8
export MAX_NUM_ITERATIVE_REASONER_CALLS=10
export USE_GEMINI_AS_TOOL="False"

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    echo "Starting evaluation on GPU $IDX..."
    CUDA_VISIBLE_DEVICES=$IDX python sage/eval/process.py \
        --model_name qwen3_vl:$MODEL \
        --benchmark $BENCHMARK \
        --workers 1 \
        --num_gpus $NUM_GPUS \
        --gpu_idx $IDX \
        --max_num_iterative_reasoner_calls $MAX_NUM_ITERATIVE_REASONER_CALLS \
        --use_gemini_as_tool $USE_GEMINI_AS_TOOL &
done

wait

python sage/eval/evaluate_responses.py qwen3_vl_Qwen3-VL-8B-Instruct_${BENCHMARK}_results.jsonl