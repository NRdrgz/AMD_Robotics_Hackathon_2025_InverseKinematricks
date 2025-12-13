"""Sort policy configuration for the black arm.

This policy sorts packages to the correct area based on tape color.
Based on SmolVLA with RTC disabled.
"""

from .base import PolicyConfig, PolicyType

# Default HuggingFace path for the sort policy
DEFAULT_SORT_POLICY_PATH = "giacomoran/hackathon_amd_mission2_black_sort_smolvla"

# Default task prompt
DEFAULT_SORT_TASK = (
    "Pick up the package and place it inside the taped square on the table whose color "
    "matches the tape on top of the package (red package-tape to red square, yellow "
    "package-tape to yellow square). Place the package fully within the square boundaries."
)


def create_sort_policy_config(
    hf_path: str = DEFAULT_SORT_POLICY_PATH,
    device: str = "cuda",
    task: str = DEFAULT_SORT_TASK,
) -> PolicyConfig:
    """Create configuration for the sort policy.

    Args:
        hf_path: HuggingFace model path
        device: Device to run inference on
        task: Task description for the policy

    Returns:
        PolicyConfig for the sort policy
    """
    return PolicyConfig(
        hf_path=hf_path,
        policy_type=PolicyType.SMOLVLA,
        device=device,
        rtc_enabled=False,
        rtc_execution_horizon=30,
        fps=30.0,
        task=task,
    )
