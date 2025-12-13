#!/usr/bin/env python
"""Fake arms computer for debugging WebSocket connection with conveyor belt computer.

This script simulates the arms computer behavior without any real hardware:
- No robot arms or cameras needed
- No policy loading or inference
- Just WebSocket communication for debugging

Usage:
    uv run python src/fake_main_arms.py --conveyor-host=<host>

    # With default localhost:
    uv run python src/fake_main_arms.py --conveyor-host=localhost

    # With custom port:
    uv run python src/fake_main_arms.py --conveyor-host=10.33.1.59 --conveyor-port=8765
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from communication.client import ArmsWebSocketClient
from communication.messages import SystemState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fake arms computer for debugging WebSocket connection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    parser.add_argument(
        "--ack-delay",
        type=float,
        default=0.1,
        help="Simulated delay before sending ACK (seconds)",
    )
    parser.add_argument(
        "--fail-rate",
        type=float,
        default=0.0,
        help="Rate of simulated failures (0.0 to 1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


class FakeArmController:
    """Fake arm controller that just tracks state."""

    def __init__(self, ack_delay: float = 0.1, fail_rate: float = 0.0):
        """Initialize fake arm controller.

        Args:
            ack_delay: Delay before acknowledging state changes (simulates processing)
            fail_rate: Rate of simulated failures (0.0 to 1.0)
        """
        self._state = SystemState.STOPPED
        self._ack_delay = ack_delay
        self._fail_rate = fail_rate
        self._state_history: list[tuple[str, SystemState]] = []

    def get_state(self) -> SystemState:
        """Get current state."""
        return self._state

    async def set_state(self, state: SystemState) -> bool:
        """Set new state with simulated delay.

        Args:
            state: New state to set

        Returns:
            True if successful, False if simulated failure
        """
        import random

        old_state = self._state

        # Simulate processing delay
        if self._ack_delay > 0:
            await asyncio.sleep(self._ack_delay)

        # Simulate random failures if configured
        if self._fail_rate > 0 and random.random() < self._fail_rate:
            logger.warning(
                f"🔴 SIMULATED FAILURE: State change {old_state.value} -> {state.value}"
            )
            return False

        self._state = state
        self._state_history.append((f"{old_state.value} -> {state.value}", state))

        # Log with nice formatting
        state_emoji = {
            SystemState.STOPPED: "⏹️",
            SystemState.RUNNING: "▶️",
            SystemState.FLIPPING: "🔄",
            SystemState.SORTING: "📦",
        }
        emoji = state_emoji.get(state, "❓")
        logger.info(f"{emoji} State changed: {old_state.value} -> {state.value}")

        return True

    def print_history(self) -> None:
        """Print state change history."""
        logger.info("=" * 50)
        logger.info("State change history:")
        for i, (transition, _) in enumerate(self._state_history, 1):
            logger.info(f"  {i}. {transition}")
        logger.info("=" * 50)


async def run_fake_arms(args: argparse.Namespace) -> None:
    """Main async function for fake arms computer.

    Args:
        args: Parsed command line arguments
    """
    shutdown_event = asyncio.Event()

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: shutdown_event.set())

    # Create fake arm controller
    arm_controller = FakeArmController(
        ack_delay=args.ack_delay,
        fail_rate=args.fail_rate,
    )

    ws_client: ArmsWebSocketClient | None = None

    try:
        # Create WebSocket client
        ws_client = ArmsWebSocketClient(
            host=args.conveyor_host,
            port=args.conveyor_port,
            on_state_change=None,  # Will be set below
            on_connected=None,
            on_disconnected=None,
        )

        # Set callbacks
        async def on_state_change(state: SystemState) -> bool:
            logger.info(f"📡 Received state change request: {state.value}")
            return await arm_controller.set_state(state)

        async def on_connected() -> None:
            current = arm_controller.get_state()
            if current == SystemState.STOPPED:
                logger.info("🔌 Connected! Transitioning from STOPPED to RUNNING")
                await arm_controller.set_state(SystemState.RUNNING)
            else:
                logger.info(f"🔌 Connected! (current state: {current.value})")

        async def on_disconnected() -> None:
            current = arm_controller.get_state()
            if current != SystemState.STOPPED:
                logger.warning(
                    f"🔌 Disconnected! Transitioning from {current.value} to STOPPED"
                )
                await arm_controller.set_state(SystemState.STOPPED)

        ws_client.on_state_change = on_state_change
        ws_client.on_connected = on_connected
        ws_client.on_disconnected = on_disconnected

        # Start WebSocket client
        await ws_client.start()
        logger.info(f"🔌 Connecting to conveyor computer at {ws_client.uri}...")

        # Wait for connection
        if await ws_client.wait_for_connection(timeout=30.0):
            logger.info("✅ Connected to conveyor computer")
        else:
            logger.warning(
                "⚠️  Could not connect to conveyor computer, will keep trying..."
            )

        # Main loop - just wait for shutdown
        logger.info("=" * 50)
        logger.info("Fake arms computer running. Press Ctrl+C to stop.")
        logger.info("=" * 50)

        while not shutdown_event.is_set():
            await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        logger.info("Shutting down...")

        if ws_client:
            try:
                await asyncio.wait_for(ws_client.stop(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket client stop timed out")
            except Exception as e:
                logger.error(f"Error stopping WebSocket client: {e}")

        # Print state history
        arm_controller.print_history()

        logger.info("Fake arms computer shutdown complete")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("  FAKE ARMS COMPUTER - WebSocket Connection Debug")
    logger.info("=" * 60)
    logger.info(f"  Conveyor host: {args.conveyor_host}:{args.conveyor_port}")
    logger.info(f"  ACK delay: {args.ack_delay}s")
    logger.info(f"  Fail rate: {args.fail_rate * 100:.1f}%")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  This is a FAKE arms computer for debugging.")
    logger.info("  No real hardware is used - just WebSocket communication.")
    logger.info("")
    logger.info("=" * 60)

    try:
        asyncio.run(run_fake_arms(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
