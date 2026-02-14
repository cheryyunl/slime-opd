"""
On-Policy Distillation reward function for Vision-Language Models (VLM).

This extends the standard OPD reward function to support multimodal inputs (images).
Based on examples/on_policy_distillation/on_policy_distillation.py with VLM support added.
"""

import aiohttp
import torch

from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    """
    OPD reward function for VLM - calls teacher model to get log-probs.

    Extends the text-only version by adding image_data to the payload when present.
    """
    payload = {
        "input_ids": sample.tokens,  # All tokens: prompt + student's response
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,  # Don't generate, just get log-probs
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,  # Return log-probs for all positions
    }

    # VLM-specific: Add images to payload if present
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        from slime.utils.processing_utils import encode_image_for_rollout_engine
        payload["image_data"] = [
            encode_image_for_rollout_engine(img)
            for img in sample.multimodal_inputs["images"]
        ]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """
    Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (sglang's logprob output)
    2. Trims them to match the response length (only generated tokens, not prompt/image)
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs
    for ALL tokens (image tokens + prompt + response). We only extract the response portion.

    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.

    For hybrid OPD + task rewards, you can modify scalar_rewards below to include task scores.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32
        )
        for reward in raw_rewards
    ]

    # Trim to only the response tokens (last N tokens)
    # The full sequence is: [image_tokens, prompt_tokens, response_tokens]
    # We only want log-probs for response_tokens for OPD
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    # Store teacher log-probs in samples for OPD KL computation
    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.

    # OPTION 1: Pure distillation (no task reward)
    scalar_rewards = [0.0] * len(samples)

    # OPTION 2: Hybrid OPD + task reward (uncomment to use)
    # You can compute task rewards (e.g., math correctness) and add them:
    # from slime.rollout.rm_hub.math_rm import compute_math_reward
    # task_rewards = [compute_math_reward(s) for s in samples]
    # scalar_rewards = task_rewards  # Or blend: 0.5 * task + 0.5 * opd_kl

    return scalar_rewards, scalar_rewards
