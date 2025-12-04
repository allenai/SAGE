#!/bin/bash

export SERPER_API_KEY="YOUR_SERPER_API_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

export TOOL_CALL_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
export VLLM_CLIENT_URL="vLLM_API_URL_FOR_TOOL_CALLING"
export TRANSCRIBE_API_URL="API_URL_FOR_TRANSCRIPTION"
export VIDEO_DIR="data/sage_bench_videos"

export MODEL=$1
export BENCHMARK="sage_bench" # minerva_bench
export NUM_GPUS=$2
export MAX_NUM_ITERATIVE_REASONER_CALLS=10
export USE_GEMINI_AS_TOOL="False"

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    echo "Starting evaluation on GPU $IDX..."
    CUDA_VISIBLE_DEVICES=$IDX python sage/eval/process.py \
        --model_name sage:$MODEL \
        --benchmark $BENCHMARK \
        --workers 1 \
        --num_gpus $NUM_GPUS \
        --gpu_idx $IDX \
        --max_num_iterative_reasoner_calls $MAX_NUM_ITERATIVE_REASONER_CALLS \
        --use_gemini_as_tool $USE_GEMINI_AS_TOOL &
done

wait

python sage/eval/evaluate_responses.py sage_$3_${BENCHMARK}_results.jsonl