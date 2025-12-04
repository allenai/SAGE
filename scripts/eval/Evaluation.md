# Evaluation on SAGE-Bench

>Note: For transparency, we share the JSONL and JSON files generated during evaluation for all models in the [HF Hub Collection](https://huggingface.co/collections/allenai/sage) under the [`results/`](../../results/) directory

- Download the YouTube videos into `data/sage_bench_videos` using the video IDs listed in [allenai/SAGE-Bench](https://huggingface.co/datasets/allenai/SAGE-Bench).

- Run the evaluation for the SAGE-MM-Qwen3-VL-8B-SFT_RL model as the orchestrator:

    ```bash
    # set the environment variables at top of scripts/eval/eval_sage.sh
    export SERPER_API_KEY="YOUR_SERPER_API_KEY"
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

    export TOOL_CALL_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
    export VLLM_CLIENT_URL="vLLM_API_URL_FOR_TOOL_CALLING"
    export TRANSCRIBE_API_URL="API_URL_FOR_TRANSCRIPTION"
    
    bash scripts/eval/eval_sage.sh allenai/SAGE-MM-Qwen3-VL-8B-SFT_RL <NUM-GPUS> SAGE-MM-Qwen3-VL-8B-SFT_RL
    ```

    The eval results will be saved to `sage_SAGE-MM-Qwen3-VL-8B-SFT_RL_sage_bench_results.json` file with full tool call traces saved inside `sage_SAGE-MM-Qwen3-VL-8B-SFT_RL_sage_bench_results.jsonl`.

- If you want to evaluate any Molmo2 based models, please downgrade vllm to 0.10.2 before running any commands:

    ```bash
    pip install vllm==0.10.2
    
    # set the environment variables at top of scripts/eval/eval_sage.sh
    ...
    
    bash scripts/eval/eval_sage.sh allenai/SAGE-MM-Molmo2-8B-SFT_RL <NUM-GPUS> SAGE-MM-Molmo2-8B-SFT_RL
    ```