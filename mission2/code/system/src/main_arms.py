#!/usr/bin/env python
"""Main entry point for the arms computer.

This script:
1. Loads all three policies (pick, flip, sort)
2. Connects to both robot arms (blue and black)
3. Connects to the conveyor belt computer via WebSocket
4. Manages state transitions based on commands from the conveyor computer

Usage:
    python src/main_arms.py \
        --blue-arm-port=/dev/ttyACM3 \
        --black-arm-port=/dev/ttyACM1 \
        --blue-top-camera=/dev/video8 \
        --blue-wrist-camera=/dev/video6 \
        --black-top-camera=/dev/video4 \
        --black-wrist-camera=/dev/video2 \
        --blue-arm-id=blue_follower \
        --black-arm-id=black_follower \
        --conveyor-host=10.33.1.59 \
        --device=cuda
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from threading import Event

# Configure logging BEFORE any lerobot imports to avoid lerobot's logger config interfering
# Use force=True to prevent lerobot from overriding our logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Force reconfiguration even if logging was already configured
)

from arm_controller import ArmController
from communication.client import ArmsWebSocketClient
from communication.messages import SystemState
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from policies.base import PolicyWrapper
from policies.flip import create_flip_policy_config
from policies.pick import create_pick_policy_config
from policies.sort import create_sort_policy_config

# Re-assert our logging configuration after lerobot imports to ensure it's not overridden
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Remove any handlers lerobot might have added
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
# Add our own handler if none exists
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(handler)

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
        default="giacomoran/hackathon_amd_mission2_blue_pick",
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
        default="giacomoran/hackathon_amd_mission2_black_sort",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
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
        ACT policies use the original camera names.
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
    current_state = arm_controller.get_state()
    logger.info(
        f"Received state change request: {current_state.value} -> {state.value}"
    )
    try:
        result = arm_controller.set_state(state)
        if result:
            logger.info(f"Successfully changed state to {state.value}")
        else:
            logger.warning(f"State change to {state.value} returned False")
        return result
    except Exception as e:
        logger.error(f"Failed to change state to {state.value}: {e}", exc_info=True)
        return False


async def run_arms_computer(args: argparse.Namespace) -> None:
    """Main async function for the arms computer.

    Args:
        args: Parsed command line arguments
    """
    shutdown_event = Event()

    # Setup signal handlers using asyncio-compatible method
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Fallback for platforms that don't support add_signal_handler
        logger.warning("Signal handlers not supported on this platform, using fallback")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: shutdown_event.set())

    arm_controller: ArmController | None = None
    ws_client: ArmsWebSocketClient | None = None

    try:
        # Load policies
        logger.info("=" * 50)
        logger.info("Loading policies...")
        logger.info("=" * 50)

        logger.info(f"Loading PICK policy from: {args.pick_policy_path}")
        pick_config = create_pick_policy_config(
            hf_path=args.pick_policy_path,
            device=args.device,
        )
        pick_policy = PolicyWrapper(pick_config)
        pick_policy.load()
        logger.info(f"✅ PICK policy loaded successfully (device: {args.device})")

        logger.info(f"Loading FLIP policy from: {args.flip_policy_path}")
        flip_config = create_flip_policy_config(
            hf_path=args.flip_policy_path,
            device=args.device,
        )
        flip_policy = PolicyWrapper(flip_config)
        flip_policy.load()
        logger.info(f"✅ FLIP policy loaded successfully (device: {args.device})")

        logger.info(f"Loading SORT policy from: {args.sort_policy_path}")
        sort_config = create_sort_policy_config(
            hf_path=args.sort_policy_path,
            device=args.device,
        )
        sort_policy = PolicyWrapper(sort_config)
        sort_policy.load()
        logger.info(f"✅ SORT policy loaded successfully (device: {args.device})")

        logger.info("=" * 50)
        logger.info("All policies loaded successfully")
        logger.info("=" * 50)

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
            shutdown_event=shutdown_event,
        )

        # Connect to robots
        arm_controller.connect()

        # Start arm controller
        arm_controller.start()

        # Create WebSocket client
        async def on_state_change(state: SystemState) -> bool:
            logger.info(
                f"📡 WebSocket received state change notification: {state.value}"
            )
            return await handle_state_change(state, arm_controller)

        ws_client = ArmsWebSocketClient(
            host=args.conveyor_host,
            port=args.conveyor_port,
            on_state_change=on_state_change,
        )

        # Start WebSocket client
        await ws_client.start()
        logger.info(f"🔌 Connecting to conveyor computer at {ws_client.uri}...")

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
            # Sleep in small chunks to check shutdown more frequently
            for _ in range(10):  # 10 * 0.01s = 0.1s total
                if shutdown_event.is_set():
                    break
                await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Shutting down...")

        # Stop arm controller first (stops all threads)
        if arm_controller:
            try:
                arm_controller.stop()
                logger.info("Waiting for threads to stop...")
                # Give threads time to stop
                import time as time_module

                time_module.sleep(1.0)
            except Exception as e:
                logger.error(f"Error stopping arm controller: {e}")

        # Stop WebSocket client
        if ws_client:
            try:
                await asyncio.wait_for(ws_client.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket client stop timed out")
            except Exception as e:
                logger.error(f"Error stopping WebSocket client: {e}")

        # Disconnect robots last
        if arm_controller:
            try:
                arm_controller.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting robots: {e}")

        logger.info("Arms computer shutdown complete")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

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
