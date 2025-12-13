"""Flip policy configuration for the black arm.

This policy flips packages on the conveyor belt to expose the barcode.
Based on ACT (no RTC).
"""

from .base import PolicyConfig, PolicyType

# Default HuggingFace path for the flip policy
DEFAULT_FLIP_POLICY_PATH = "giacomoran/hackathon_amd_mission2_black_flip_act"

# Default task prompt
DEFAULT_FLIP_TASK = (
    "Flip the package over on the conveyor belt so that the colored tape is visible, "
    "then place it back in its original position."
)


def create_flip_policy_config(
    hf_path: str = DEFAULT_FLIP_POLICY_PATH,
    device: str = "cuda",
    task: str = DEFAULT_FLIP_TASK,
) -> PolicyConfig:
    """Create configuration for the flip policy.

    Args:
        hf_path: HuggingFace model path
        device: Device to run inference on
        task: Task description for the policy

    Returns:
        PolicyConfig for the flip policy
    """
    return PolicyConfig(
        hf_path=hf_path,
        policy_type=PolicyType.ACT,
        device=device,
        rtc_enabled=False,  # ACT doesn't use RTC
        fps=30.0,
        task=task,
    )

