"""Base policy wrapper for loading and running robot policies.

Supports both SmolVLA (with RTC) and ACT policies.
Based on the pattern from eval_with_real_robot.py.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from torch import Tensor

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """Supported policy types."""

    SMOLVLA = "smolvla"
    ACT = "act"
    PI0 = "pi0"
    PI05 = "pi05"


@dataclass
class PolicyConfig:
    """Configuration for a policy."""

    hf_path: str
    policy_type: PolicyType
    device: str = "cuda"
    rtc_enabled: bool = True
    rtc_execution_horizon: int = 30
    rtc_max_guidance_weight: float = 1.0
    fps: float = 30.0
    # Request new actions when queue has this many or fewer actions remaining
    # Lower values = more responsive but more frequent inference calls
    # At 30 FPS with ~100ms inference, 5-10 gives ~0.15-0.3s buffer
    action_queue_size_to_get_new_actions: int = 10
    task: str = ""


class PolicyWrapper:
    """Wrapper for loading and running robot policies.

    Handles both SmolVLA (with RTC) and ACT policies with a unified interface.
    """

    def __init__(self, config: PolicyConfig):
        """Initialize the policy wrapper.

        Args:
            config: Policy configuration
        """
        self.config = config
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._loaded = False

        # RTC components (for SmolVLA/Pi0)
        self._rtc_config: RTCConfig | None = None
        self._action_queue: ActionQueue | None = None
        self._latency_tracker: LatencyTracker | None = None

        # ACT components
        self._act_action_buffer: list[Tensor] = []
        self._act_action_index: int = 0

    @property
    def is_loaded(self) -> bool:
        """Check if the policy is loaded."""
        return self._loaded

    @property
    def uses_rtc(self) -> bool:
        """Check if this policy uses RTC."""
        return self.config.rtc_enabled and self.config.policy_type in (
            PolicyType.SMOLVLA,
            PolicyType.PI0,
            PolicyType.PI05,
        )

    def load(self) -> None:
        """Load the policy from HuggingFace Hub."""
        if self._loaded:
            logger.warning(f"Policy {self.config.hf_path} already loaded")
            return

        logger.info(f"Loading policy from {self.config.hf_path}")

        # Get policy class and load
        policy_class = get_policy_class(self.config.policy_type.value)
        pretrained_config = PreTrainedConfig.from_pretrained(self.config.hf_path)

        self._policy = policy_class.from_pretrained(
            self.config.hf_path, config=pretrained_config
        )

        # Setup RTC if applicable
        if self.uses_rtc:
            self._rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=self.config.rtc_execution_horizon,
                max_guidance_weight=self.config.rtc_max_guidance_weight,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            )
            self._policy.config.rtc_config = self._rtc_config
            self._policy.init_rtc_processor()
            self._action_queue = ActionQueue(self._rtc_config)
            self._latency_tracker = LatencyTracker()
            logger.info(
                f"RTC enabled with execution_horizon={self.config.rtc_execution_horizon}"
            )

        # Move to device
        self._policy = self._policy.to(self.config.device)
        self._policy.eval()

        # Load preprocessor and postprocessor
        # Create a mock PreTrainedConfig with required fields for make_pre_post_processors
        policy_cfg = PreTrainedConfig.from_pretrained(self.config.hf_path)
        policy_cfg.pretrained_path = self.config.hf_path

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=self.config.hf_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": self.config.device},
            },
        )

        self._loaded = True
        logger.info(f"Policy {self.config.hf_path} loaded successfully")

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        if self.uses_rtc and self._action_queue:
            self._action_queue = ActionQueue(self._rtc_config)
            self._latency_tracker = LatencyTracker()
        else:
            self._act_action_buffer = []
            self._act_action_index = 0

    def get_action(
        self,
        observation: dict[str, Tensor],
        observation_features: list[str],
    ) -> Tensor | None:
        """Get the next action from the policy.

        For RTC policies: returns the next action from the action queue,
        requesting new chunks as needed.

        For ACT policies: returns the next action from the current chunk,
        requesting new chunks when buffer is exhausted.

        Args:
            observation: Current observation dictionary
            observation_features: List of observation feature names

        Returns:
            Action tensor, or None if no action available
        """
        if not self._loaded:
            raise RuntimeError("Policy not loaded. Call load() first.")

        if self.uses_rtc:
            return self._get_action_rtc(observation, observation_features)
        else:
            return self._get_action_act(observation, observation_features)

    def _prepare_observation(
        self,
        observation: dict[str, Tensor],
        observation_features: list[str],
    ) -> dict[str, Tensor]:
        """Prepare observation for policy inference.

        Args:
            observation: Raw observation dictionary
            observation_features: List of observation feature names

        Returns:
            Processed observation ready for policy
        """
        # Convert observation features to dataset features format
        # This matches the pattern in eval_with_real_robot.py
        dataset_features = hw_to_dataset_features(observation_features, "observation")

        obs_with_policy_features = build_dataset_frame(
            dataset_features, observation, prefix="observation"
        )

        for name in obs_with_policy_features:
            obs_with_policy_features[name] = torch.from_numpy(
                obs_with_policy_features[name]
            )
            if "image" in name:
                obs_with_policy_features[name] = (
                    obs_with_policy_features[name].type(torch.float32) / 255
                )
                obs_with_policy_features[name] = (
                    obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                )
            obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
            obs_with_policy_features[name] = obs_with_policy_features[name].to(
                self.config.device
            )

        obs_with_policy_features["task"] = [self.config.task]
        obs_with_policy_features["robot_type"] = ""

        return self._preprocessor(obs_with_policy_features)

    def _get_action_rtc(
        self,
        observation: dict[str, Tensor],
        observation_features: list[str],
    ) -> Tensor | None:
        """Get action using RTC (for SmolVLA/Pi0 policies).

        Args:
            observation: Current observation dictionary
            observation_features: List of observation feature names

        Returns:
            Action tensor, or None if no action available
        """
        fps = self.config.fps
        time_per_chunk = 1.0 / fps
        get_actions_threshold = self.config.action_queue_size_to_get_new_actions

        # Check if we need to request new actions
        if self._action_queue.qsize() <= get_actions_threshold:
            current_time = time.perf_counter()
            action_index_before_inference = self._action_queue.get_action_index()
            prev_actions = self._action_queue.get_left_over()

            inference_latency = self._latency_tracker.max()
            inference_delay = (
                math.ceil(inference_latency / time_per_chunk)
                if inference_latency > 0
                else 0
            )

            # Prepare observation
            preprocessed_obs = self._prepare_observation(
                observation, observation_features
            )

            # Generate actions with RTC
            actions = self._policy.predict_action_chunk(
                preprocessed_obs,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
            )

            # Store original actions for RTC
            original_actions = actions.squeeze(0).clone()

            # Postprocess actions
            postprocessed_actions = self._postprocessor(actions)
            postprocessed_actions = postprocessed_actions.squeeze(0)

            # Track latency
            new_latency = time.perf_counter() - current_time
            new_delay = math.ceil(new_latency / time_per_chunk)
            self._latency_tracker.add(new_latency)

            # Merge into action queue
            self._action_queue.merge(
                original_actions,
                postprocessed_actions,
                new_delay,
                action_index_before_inference,
            )

        # Get next action from queue
        return self._action_queue.get()

    def _get_action_act(
        self,
        observation: dict[str, Tensor],
        observation_features: list[str],
    ) -> Tensor | None:
        """Get action using ACT-style sequential execution.

        Args:
            observation: Current observation dictionary
            observation_features: List of observation feature names

        Returns:
            Action tensor, or None if no action available
        """
        # If buffer is empty or exhausted, get new chunk
        if self._act_action_index >= len(self._act_action_buffer):
            # Prepare observation
            preprocessed_obs = self._prepare_observation(
                observation, observation_features
            )

            # Generate action chunk
            with torch.no_grad():
                actions = self._policy.select_action(preprocessed_obs)

            # Postprocess
            postprocessed_actions = self._postprocessor(actions)
            postprocessed_actions = postprocessed_actions.squeeze(0)

            # Store in buffer
            if postprocessed_actions.dim() == 1:
                # Single action
                self._act_action_buffer = [postprocessed_actions]
            else:
                # Action chunk - split into list
                self._act_action_buffer = [
                    postprocessed_actions[i]
                    for i in range(postprocessed_actions.shape[0])
                ]
            self._act_action_index = 0

        # Return next action from buffer
        if self._act_action_index < len(self._act_action_buffer):
            action = self._act_action_buffer[self._act_action_index]
            self._act_action_index += 1
            return action

        return None

    def has_pending_actions(self) -> bool:
        """Check if there are pending actions to execute.

        Returns:
            True if there are actions waiting to be executed
        """
        if self.uses_rtc:
            return self._action_queue is not None and self._action_queue.qsize() > 0
        else:
            return self._act_action_index < len(self._act_action_buffer)
