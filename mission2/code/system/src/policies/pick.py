"""Pick policy configuration for the blue arm.

This policy picks packages and places them on the conveyor belt.
Based on ACT.
"""

from .base import PolicyConfig, PolicyType

# Default HuggingFace path for the pick policy
DEFAULT_PICK_POLICY_PATH = (
    "giacomoran/hackathon_amd_mission2_blue_pick_act_cardboard_v2"
)

# Default task prompt
DEFAULT_PICK_TASK = "Pick up the package and place it on the black conveyor belt."


def create_pick_policy_config(
    hf_path: str = DEFAULT_PICK_POLICY_PATH,
    device: str = "cuda",
    task: str = DEFAULT_PICK_TASK,
) -> PolicyConfig:
    """Create configuration for the pick policy.

    Args:
        hf_path: HuggingFace model path
        device: Device to run inference on
        task: Task description for the policy

    Returns:
        PolicyConfig for the pick policy
    """
    return PolicyConfig(
        hf_path=hf_path,
        policy_type=PolicyType.ACT,
        device=device,
        rtc_enabled=False,  # ACT does not support RTC
        fps=30.0,
        task=task,
    )
