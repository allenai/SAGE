# Training SAGE-MM

## SFT Stage

- Download the YouTube videos into `data/videos` using the video IDs listed in [allenai/SAGE-MM-SFT-417K](https://huggingface.co/datasets/allenai/SAGE-MM-SFT-417K).

- Run SFT on 8 GPUs:
    ```bash
    # Qwen2.5/3-VL based models
    bash scripts/train/sft.sh

    # Molmo2 based models
    bash scripts/train/sft_molmo.sh
    ```

## RL Stage

>Note: You can directly use the SFT checkpoints available on the [HF Hub collection](https://huggingface.co/collections/allenai/sage) for training with RL.

- Download the YouTube videos into `data/videos` using the video IDs listed in [allenai/SAGE-MM-RL-7K](https://huggingface.co/datasets/allenai/SAGE-MM-RL-7K).

- Train on 8 GPUs

    ```bash
    # Qwen2.5/3-VL based models

    # set the env variables at the top of the script
    export SERPER_API_KEY="YOUR_SERPER_API_KEY"
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

    export TOOL_CALL_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
    export VLLM_API_URL="vLLM_API_URL_FOR_TOOL_CALLING"
    export TRANSCRIBE_API_URL="API_URL_FOR_TRANSCRIPTION"

    bash scripts/train/grpo.sh

    # Molmo2 based models

    pip install vllm==0.10.2
    # set the env variables at the top of the script
    ...
    bash scripts/train/grpo_molmo.sh
    ```