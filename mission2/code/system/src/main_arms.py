#!/usr/bin/env python
"""Main entry point for the arms computer.

This script:
1. Loads all three policies (pick, flip, sort)
2. Connects to both robot arms (blue and black)
3. Connects to the conveyor belt computer via WebSocket
4. Manages state transitions based on commands from the conveyor computer

Usage:
    python src/main_arms.py \
        --blue-arm-port=/dev/ttyACM0 \
        --black-arm-port=/dev/ttyACM1 \
        --blue-top-camera=/dev/video6 \
        --blue-wrist-camera=/dev/video8 \
        --black-top-camera=/dev/video4 \
        --black-wrist-camera=/dev/video2 \
        --conveyor-host=100.86.200.31 \
        --device=cuda
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from threading import Event

from arm_controller import ArmController
from communication.client import ArmsWebSocketClient
from communication.messages import SystemState
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from policies.base import PolicyWrapper
from policies.flip import create_flip_policy_config
from policies.pick import create_pick_policy_config
from policies.sort import create_sort_policy_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Arms computer for package sorting system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Robot configuration
    parser.add_argument(
        "--blue-arm-port",
        type=str,
        required=True,
        help="Serial port for the blue arm",
    )
    parser.add_argument(
        "--black-arm-port",
        type=str,
        required=True,
        help="Serial port for the black arm",
    )
    parser.add_argument(
        "--blue-arm-id",
        type=str,
        default="blue_arm",
        help="ID for the blue arm",
    )
    parser.add_argument(
        "--black-arm-id",
        type=str,
        default="black_arm",
        help="ID for the black arm",
    )

    # Camera configuration for blue arm
    parser.add_argument(
        "--blue-top-camera",
        type=str,
        default="/dev/video6",
        help="Camera path for blue arm top camera",
    )
    parser.add_argument(
        "--blue-wrist-camera",
        type=str,
        default="/dev/video8",
        help="Camera path for blue arm wrist camera",
    )

    # Camera configuration for black arm
    parser.add_argument(
        "--black-top-camera",
        type=str,
        default="/dev/video4",
        help="Camera path for black arm top camera",
    )
    parser.add_argument(
        "--black-wrist-camera",
        type=str,
        default="/dev/video2",
        help="Camera path for black arm wrist camera",
    )

    # Camera settings (shared)
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera frame width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera frame height",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Camera frames per second",
    )

    # Policy configuration
    parser.add_argument(
        "--pick-policy-path",
        type=str,
        default="giacomoran/hackathon_amd_mission2_blue_pick_smolvla",
        help="HuggingFace path for pick policy",
    )
    parser.add_argument(
        "--flip-policy-path",
        type=str,
        default="giacomoran/hackathon_amd_mission2_black_flip_act",
        help="HuggingFace path for flip policy",
    )
    parser.add_argument(
        "--sort-policy-path",
        type=str,
        default="giacomoran/hackathon_amd_mission2_black_sort_smolvla",
        help="HuggingFace path for sort policy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for policy inference (cuda, mps, cpu)",
    )

    # WebSocket configuration
    parser.add_argument(
        "--conveyor-host",
        type=str,
        required=True,
        help="Host address of the conveyor belt computer",
    )
    parser.add_argument(
        "--conveyor-port",
        type=int,
        default=8765,
        help="WebSocket port of the conveyor belt computer",
    )

    # Execution configuration
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Action execution frequency in Hz",
    )

    return parser.parse_args()


def create_robot_config(
    port: str,
    robot_id: str,
    top_camera_path: str,
    wrist_camera_path: str,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> SO101FollowerConfig:
    """Create a robot configuration.

    Args:
        port: Serial port for the robot
        robot_id: ID for the robot
        top_camera_path: Path for the top camera (e.g., /dev/video4)
        wrist_camera_path: Path for the wrist camera (e.g., /dev/video2)
        camera_width: Camera frame width
        camera_height: Camera frame height
        camera_fps: Camera FPS

    Returns:
        SO101FollowerConfig instance

    Note:
        Camera names 'top' and 'wrist' are used by the robot.
        SmolVLA policies expect 'camera1' (top) and 'camera2' (wrist).
        ACT policies use the original names.
    """
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=top_camera_path,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_camera_path,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        ),
    }

    return SO101FollowerConfig(
        port=port,
        id=robot_id,
        cameras=cameras,
    )


async def handle_state_change(
    state: SystemState,
    arm_controller: ArmController,
) -> bool:
    """Handle a state change command from the conveyor computer.

    Args:
        state: New system state
        arm_controller: Arm controller instance

    Returns:
        True if state change was successful
    """
    try:
        return arm_controller.set_state(state)
    except Exception as e:
        logger.error(f"Failed to change state to {state.value}: {e}")
        return False


async def run_arms_computer(args: argparse.Namespace) -> None:
    """Main async function for the arms computer.

    Args:
        args: Parsed command line arguments
    """
    shutdown_event = Event()

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    arm_controller: ArmController | None = None
    ws_client: ArmsWebSocketClient | None = None

    try:
        # Load policies
        logger.info("Loading policies...")

        pick_config = create_pick_policy_config(
            hf_path=args.pick_policy_path,
            device=args.device,
        )
        pick_policy = PolicyWrapper(pick_config)
        pick_policy.load()

        flip_config = create_flip_policy_config(
            hf_path=args.flip_policy_path,
            device=args.device,
        )
        flip_policy = PolicyWrapper(flip_config)
        flip_policy.load()

        sort_config = create_sort_policy_config(
            hf_path=args.sort_policy_path,
            device=args.device,
        )
        sort_policy = PolicyWrapper(sort_config)
        sort_policy.load()

        logger.info("All policies loaded successfully")

        # Create robot configurations
        logger.info("Creating robot configurations...")

        blue_arm_config = create_robot_config(
            port=args.blue_arm_port,
            robot_id=args.blue_arm_id,
            top_camera_path=args.blue_top_camera,
            wrist_camera_path=args.blue_wrist_camera,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )

        black_arm_config = create_robot_config(
            port=args.black_arm_port,
            robot_id=args.black_arm_id,
            top_camera_path=args.black_top_camera,
            wrist_camera_path=args.black_wrist_camera,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )

        # Create arm controller
        logger.info("Creating arm controller...")
        arm_controller = ArmController(
            blue_arm_config=blue_arm_config,
            black_arm_config=black_arm_config,
            pick_policy=pick_policy,
            flip_policy=flip_policy,
            sort_policy=sort_policy,
            fps=args.fps,
        )

        # Connect to robots
        arm_controller.connect()

        # Start arm controller
        arm_controller.start()

        # Create WebSocket client
        async def on_state_change(state: SystemState) -> bool:
            return await handle_state_change(state, arm_controller)

        ws_client = ArmsWebSocketClient(
            host=args.conveyor_host,
            port=args.conveyor_port,
            on_state_change=on_state_change,
        )

        # Start WebSocket client
        await ws_client.start()
        logger.info(f"Connecting to conveyor computer at {ws_client.uri}...")

        # Wait for connection
        if await ws_client.wait_for_connection(timeout=30.0):
            logger.info("Connected to conveyor computer")
        else:
            logger.warning(
                "Could not connect to conveyor computer, will keep trying..."
            )

        # Main loop - just wait for shutdown
        logger.info("Arms computer running. Press Ctrl+C to stop.")
        while not shutdown_event.is_set():
            await asyncio.sleep(1.0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Shutting down...")

        if ws_client:
            await ws_client.stop()

        if arm_controller:
            arm_controller.stop()
            arm_controller.disconnect()

        logger.info("Arms computer shutdown complete")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("  ARMS COMPUTER - Package Sorting System")
    logger.info("=" * 50)
    logger.info(f"  Blue arm port: {args.blue_arm_port}")
    logger.info(f"  Black arm port: {args.black_arm_port}")
    logger.info(f"  Conveyor host: {args.conveyor_host}:{args.conveyor_port}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 50)

    try:
        asyncio.run(run_arms_computer(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
