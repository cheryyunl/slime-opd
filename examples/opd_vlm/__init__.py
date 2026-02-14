"""
On-Policy Distillation for Vision-Language Models

This module provides OPD support for VLM training on GEO3K dataset.
"""

from .opd_vlm_reward import reward_func, post_process_rewards

__all__ = ["reward_func", "post_process_rewards"]
