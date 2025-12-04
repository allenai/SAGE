# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from rl_train.agent.utils_agent import to_list_of_dict, torch_to_numpy
from typing import List, Dict
from rl_train.agent.env import SAGE_RL_Environment
from sage.main import SAGE
from collections import defaultdict
from tqdm import tqdm
import os
import json
import time

IS_MOLMO2 = os.environ.get("IS_MOLMO2", "False")
IS_MOLMO2 = IS_MOLMO2.lower() == "true"
USE_GEMINI_AS_TOOL = os.environ.get("USE_GEMINI_AS_TOOL", "False")
USE_GEMINI_AS_TOOL = USE_GEMINI_AS_TOOL.lower() == "true"

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
            enable_profiling: Whether to enable trajectory collection profiling
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.sage_env = SAGE_RL_Environment(processor=processor, tokenizer=tokenizer, config=config)
        self.reward_fn_types = self.sage_env.reward_fns
        if hasattr(self.config.actor_rollout_ref.agent, "vlm_api_type"):
            self.vlm_api_type = self.config.actor_rollout_ref.agent.vlm_api_type
        else:
            self.vlm_api_type = "qwen"
    
    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs_dict: List[Dict],
    ):

        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            obs_dict,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}

        raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                                        prompt=prompt_with_chat_template,
                                        tokenizer=self.tokenizer,
                                        max_length=self.config.data.max_prompt_length,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        left_pad=True,
                                        truncation=self.config.data.truncation
                                    )
        
        
        if not IS_MOLMO2 and self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=None,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        raw_prompt = self.tokenizer.decode(raw_prompt_ids)
        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'index': item,
            "raw_prompt": raw_prompt,
        })
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: List[List[Dict]], 
    ) -> DataProto:

        if obs is None:
            return gen_batch

        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []

        placeholder_obs = None
        for ob in obs:
            if ob is not None:
                placeholder_obs = ob
                break

        # Process each sample in parallel
        for item in range(batch_size):
            # Determine if this sample uses placeholder observation
            uses_placeholder_obs = placeholder_obs is not None and (obs[item] is None or obs[item] == placeholder_obs)
            
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs_dict=placeholder_obs if obs[item] is None else obs[item],
            )
            
            processed['uses_placeholder_obs'] = uses_placeholder_obs
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            meta_info: Dict,
            all_episode_reward_dict: Dict,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            meta_info (Dict): Metadata information
            all_episode_reward_dict (Dict): All episode reward dictionary

        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)
        episode_rewards_std = np.std(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)
        episode_lengths_std = np.std(episode_lengths)

        reward_dict = {}
        for key, value in all_episode_reward_dict.items():
            reward_dict[key] = {
                "mean": np.mean(value),
                "min": np.min(value),
                "max": np.max(value),
                "std": np.std(value),
            }

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        # Track tool calls and rewards for data logging
        tool_call_counts = defaultdict(int)
        
        # Collect tool call data from all trajectories
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                if data['active_masks']:
                    # Extract tool name from the data if available
                    tool_name = data.get('tool_name', None)
                    if tool_name is not None:
                        # Clean tool name (remove any suffixes like _#1)
                        clean_tool_name = tool_name.split('_#')[0]
                        tool_call_counts[clean_tool_name] += 1
        
        # Calculate tool call metrics for data
        tool_call_metrics = {}
        
        # Add mean rewards per tool type
        for tool_name in tool_call_counts.keys():
            tool_call_metrics[f"tool_calls/{tool_name}_count"] = tool_call_counts[tool_name]
        
        # Add total tool call count
        total_tool_calls = sum(tool_call_counts.values())
        tool_call_metrics["tool_calls/total_count"] = total_tool_calls
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # reward_dict
                    
                    data["episode/rewards"] = episode_rewards[bs]
                    data["episode/rewards_max"] = episode_rewards_max
                    data["episode/rewards_min"] = episode_rewards_min
                    data["episode/rewards_std"] = episode_rewards_std
                    data["episode/lengths"] = episode_lengths[bs]
                    data["episode/lengths_min"] = episode_lengths_min
                    data["episode/lengths_max"] = episode_lengths_max
                    data["episode/lengths_std"] = episode_lengths_std
                    for key, value in reward_dict.items():
                        for k, v in value.items():
                            data[f"episode/{key}/{k}"] = v

                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    # Add tool call metrics to each data sample
                    for key, value in tool_call_metrics.items():
                        data[key] = value

                    effective_batch.append(data)
            
        data = collate_fn(effective_batch)
        data.pop("index", None)
        data.pop("uses_placeholder_obs", None)
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=data,
            meta_info=meta_info
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            global_step,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            global_step (int): Global step
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None
        
        uid = str(uuid.uuid4())
        uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)

        all_episode_reward_dict = {}
        for reward_fn_type in self.reward_fn_types:
            all_episode_reward_dict[reward_fn_type] = np.zeros(batch_size, dtype=np.float32)

        sage_agents = [
            SAGE(
                vlm_api_type=self.vlm_api_type,
                is_rl_train_mode=True,
                tool_call_clients=self.sage_env.tool_call_clients,
                use_gemini_as_tool=USE_GEMINI_AS_TOOL,
            ) for _ in range(batch_size)
        ]
        
        obs = None
        video_paths = []
        questions = []
        gt_answers = []
        video_duration_timestamps = []
        video_contexts = []
        raw_inputs = gen_batch.non_tensor_batch.pop("raw_inputs")
        assert len(raw_inputs) == batch_size, f"raw_inputs length {len(raw_inputs)} does not match batch size {batch_size}"
        for item in range(batch_size):
            video_paths.append(raw_inputs[item]['video_path'][0])
            questions.append(raw_inputs[item]['question'][0])
            gt_answers.append(raw_inputs[item]['gt_answer'][0])
            video_duration_timestamps.append(raw_inputs[item]['video_duration_timestamp'][0])

        meta_info = {
            'timing': {
                'generate_sequences': 0,
                'reshard': 0
            }
        }
        
        if global_step < self.config.actor_rollout_ref.agent.turn_scale_point:
            max_turns = self.config.actor_rollout_ref.agent.max_turns // 2
        else:
            max_turns = self.config.actor_rollout_ref.agent.max_turns
        
        max_turns += 1
        
        # Trajectory collection loop
        for _step in tqdm(range(max_turns), desc="Trajectory collection loop"):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            is_placeholder_obs = batch.non_tensor_batch.get('uses_placeholder_obs', [False] * batch_size)

            # Profile batch preparation
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = []
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "raw_prompt_ids" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt_ids")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = batch.meta_info

            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            batch.meta_info = batch_output.meta_info

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            next_obs, rewards, dones, infos, video_contexts, reward_infos = self.sage_env.step(
                text_actions, actor_rollout_wg, sage_agents, 
                video_paths, questions, gt_answers, video_duration_timestamps, 
                video_contexts=video_contexts, is_cvlm=(_step == 0), turn=_step+1, 
                max_turns=max_turns, is_placeholder_obs=is_placeholder_obs, global_step=global_step
            )
            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
            
            # Add tool_name to batch data for tracking
            if 'tool_name' in infos[0]:
                batch.non_tensor_batch['tool_name'] = np.array([info.get('tool_name', None) for info in infos], dtype=object)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            for k, v in reward_infos.items():
                all_episode_reward_dict[k] += torch_to_numpy(v) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            batch.non_tensor_batch.pop("raw_inputs", None)
            batch.non_tensor_batch.pop("raw_prompt", None)
            # batch.non_tensor_batch.pop("raw_prompt_ids", None)
            batch.non_tensor_batch.pop("multi_modal_data", None)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        self.sage_env.reset()
        
        success: Dict[str, np.ndarray] = self.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, meta_info, all_episode_reward_dict
    
    def success_evaluator(self, total_infos, total_batch_list) -> Dict[str, np.ndarray]:
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['final_answer'] is not None)
                success['success_rate'].append(won_value)
                return

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            is_train: bool = True,
            global_step: int = 0,
            ) -> DataProto:

        total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, meta_info, all_episode_reward_dict = \
            self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            global_step=global_step,
        )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            meta_info=meta_info,
            all_episode_reward_dict=all_episode_reward_dict
        )
        
        return gen_batch_output