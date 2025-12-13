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

import websockets
from communication.messages import (
    AckMessage,
    PingMessage,
    PongMessage,
    SetStateMessage,
    SystemState,
    parse_message,
    serialize_message,
)
from websockets.client import WebSocketClientProtocol

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

    Uses direct WebSocket connection with verbose logging for debugging.

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

    uri = f"ws://{args.conveyor_host}:{args.conveyor_port}"
    websocket: WebSocketClientProtocol | None = None

    async def handle_messages(ws: WebSocketClientProtocol) -> None:
        """Handle incoming WebSocket messages with verbose logging."""
        try:
            async for raw_message in ws:
                # Log the raw message
                logger.info(f"📨 RAW MESSAGE RECEIVED: {raw_message}")

                try:
                    msg = parse_message(raw_message)
                    logger.info(f"📨 PARSED MESSAGE: {type(msg).__name__} - {msg}")

                    if isinstance(msg, SetStateMessage):
                        logger.info(f"🎯 SET_STATE command received: {msg.state.value}")
                        success = await arm_controller.set_state(msg.state)

                        # Send ACK
                        ack = AckMessage(state=msg.state, success=success)
                        ack_json = serialize_message(ack)
                        logger.info(f"📤 Sending ACK: {ack_json}")
                        await ws.send(ack_json)
                    else:
                        # Handle PING with PONG
                        if isinstance(msg, PingMessage):
                            pong = PongMessage()
                            pong_json = serialize_message(pong)
                            logger.debug(f"📤 Sending PONG: {pong_json}")
                            await ws.send(pong_json)

                except ValueError as e:
                    logger.error(f"❌ Failed to parse message: {e}")
                    logger.error(f"   Raw message was: {raw_message}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔌 Connection closed: {e}")

    try:
        logger.info(f"🔌 Connecting to {uri}...")

        # Connection loop with reconnection
        while not shutdown_event.is_set():
            try:
                async with websockets.connect(uri) as ws:
                    websocket = ws
                    logger.info(f"✅ Connected to {uri}")

                    # Transition to RUNNING on connect
                    current = arm_controller.get_state()
                    if current == SystemState.STOPPED:
                        logger.info(
                            "🔌 Connected! Transitioning from STOPPED to RUNNING"
                        )
                        await arm_controller.set_state(SystemState.RUNNING)

                    logger.info("=" * 50)
                    logger.info("Fake arms computer running. Press Ctrl+C to stop.")
                    logger.info("Listening for messages...")
                    logger.info("=" * 50)

                    # Handle messages until disconnected or shutdown
                    message_task = asyncio.create_task(handle_messages(ws))

                    # Wait for either shutdown or message handler to complete
                    while not shutdown_event.is_set():
                        if message_task.done():
                            break
                        await asyncio.sleep(0.1)

                    if not message_task.done():
                        message_task.cancel()
                        try:
                            await message_task
                        except asyncio.CancelledError:
                            pass

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"🔌 Disconnected: {e}")
            except ConnectionRefusedError:
                logger.warning(f"🔌 Connection refused to {uri}")
            except Exception as e:
                logger.error(f"🔌 Connection error: {e}")

            # Transition to STOPPED on disconnect
            current = arm_controller.get_state()
            if current != SystemState.STOPPED:
                logger.warning(
                    f"🔌 Disconnected! Transitioning from {current.value} to STOPPED"
                )
                await arm_controller.set_state(SystemState.STOPPED)

            if not shutdown_event.is_set():
                logger.info("🔄 Reconnecting in 2 seconds...")
                await asyncio.sleep(2.0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        logger.info("Shutting down...")

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
