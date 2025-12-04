#!/bin/bash
set -x

export SERPER_API_KEY="YOUR_SERPER_API_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

export TOOL_CALL_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
export VLLM_API_URL="vLLM_API_URL_FOR_TOOL_CALLING"
export TRANSCRIBE_API_URL="API_URL_FOR_TRANSCRIPTION"

export VIDEO_DIR="data/videos"

export HYDRA_FULL_ERROR=1
export RAY_IGNORE_UNHANDLED_ERRORS=1

export NUM_GPUS=8
export RUN_NAME=SAGE-MM-Qwen3-VL-8B-SFT_RL
export DATASET=SAGE-MM-RL-7k/data
export SFT_CKPT=allenai/SAGE-MM-Qwen3-VL-8B-SFT
export REWARD_FNS="json-format-args_re-args-curr_web_verb_p_acc-rt"

export IS_MOLMO2="False"

export OFFLOAD_POLICY=False
export NNODES=1


python \
    rl_train/trainer/main_sage.py \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.005 \
    data.train_files=data/${DATASET}/train-00000-of-00001.parquet \
    data.val_files=data/${DATASET}/train-00000-of-00001.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=32768 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    data.shuffle=False \
    data.video_key=video_id \
    data.dataloader_num_workers=1 \
    data.return_raw_chat=True \
    reward_model.reward_manager=episode \
    +reward_model.reward_fns=${REWARD_FNS} \
    actor_rollout_ref.model.path=${SFT_CKPT} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0 \
    actor_rollout_ref.actor.optim.num_cycles=0.5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=$OFFLOAD_POLICY \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    +actor_rollout_ref.agent.turn_scale_point=100 \
    +actor_rollout_ref.agent.enable_agent=True \
    +actor_rollout_ref.agent.max_prompt_length=32768 \
    +actor_rollout_ref.agent.max_response_length=2048 \
    +actor_rollout_ref.agent.max_start_length=32768 \
    +actor_rollout_ref.agent.max_obs_length=32768 \
    +actor_rollout_ref.agent.max_turns=10 \
    +actor_rollout_ref.agent.mask_observations=True \
    +actor_rollout_ref.agent.enable_mtrl=True \
    +actor_rollout_ref.agent.max_action_length=2048 \
    +actor_rollout_ref.agent.max_concurrent_trajectories=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    trainer.logger=[console,wandb] \
    trainer.project_name=sage \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_hdfs_dir=null \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=outputs/GRPO/${RUN_NAME}/grpo_rollouts \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=20 \
    trainer.test_freq=10000000 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=outputs/GRPO/${RUN_NAME}