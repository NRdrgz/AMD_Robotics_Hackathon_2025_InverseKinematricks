#!/usr/bin/env python
"""Main entry point for the conveyor belt computer.

This script:
1. Spawns the CV Flask app as a subprocess
2. Initializes belt control via serial
3. Starts the WebSocket server for communication with the arms computer
4. Runs the state machine based on CV detection

Usage:
    uv run src/main_conveyor_belt.py \
        --belt-port=/dev/ttyUSB0 \
        --cv-app-path=../cv_classfication/src/app.py \
        --cv-api-url=http://localhost:5001 \
        --websocket-port=8765 \
        --transition-delay=2.0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import serial

# Add ramps_X_stepper to path for belt control
SCRIPT_DIR = Path(__file__).parent.resolve()
RAMPS_DIR = SCRIPT_DIR.parent.parent / "ramps_X_stepper" / "src"
sys.path.insert(0, str(RAMPS_DIR))

from belt_control import BeltControl
from communication.messages import AckMessage, SystemState
from communication.server import ConveyorWebSocketServer
from state_machine import (
    CVStatus,
    PackageSortingStateMachine,
    cv_api_status_to_cv_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Conveyor belt computer for package sorting system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Belt configuration
    parser.add_argument(
        "--belt-port",
        type=str,
        required=True,
        help="Serial port for the belt control Arduino",
    )
    parser.add_argument(
        "--belt-baudrate",
        type=int,
        default=115200,
        help="Baud rate for belt serial communication",
    )

    # CV app configuration
    parser.add_argument(
        "--cv-app-path",
        type=str,
        default="../cv_classfication/src/app.py",
        help="Path to the CV Flask app",
    )
    parser.add_argument(
        "--cv-api-url",
        type=str,
        default="http://localhost:5001",
        help="URL of the CV Flask API",
    )
    parser.add_argument(
        "--cv-startup-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for CV app startup",
    )

    # WebSocket configuration
    parser.add_argument(
        "--websocket-host",
        type=str,
        default="0.0.0.0",
        help="Host address for WebSocket server",
    )
    parser.add_argument(
        "--websocket-port",
        type=int,
        default=8765,
        help="Port for WebSocket server",
    )

    # State machine configuration
    parser.add_argument(
        "--transition-delay",
        type=float,
        default=2.0,
        help="Delay in seconds for SORTING->RUNNING transition",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Interval in seconds for polling CV API",
    )

    return parser.parse_args()


class CVAppManager:
    """Manages the CV Flask app subprocess."""

    def __init__(self, app_path: str, api_url: str, startup_timeout: float = 30.0):
        """Initialize the CV app manager.

        Args:
            app_path: Path to the CV Flask app script
            api_url: URL of the CV API
            startup_timeout: Timeout for app startup
        """
        self.app_path = Path(app_path).resolve()
        self.api_url = api_url
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None

    def start(self) -> bool:
        """Start the CV Flask app subprocess.

        Returns:
            True if app started successfully
        """
        if not self.app_path.exists():
            logger.error(f"CV app not found at {self.app_path}")
            return False

        logger.info(f"Starting CV app from {self.app_path}")

        try:
            self._process = subprocess.Popen(
                [sys.executable, str(self.app_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.app_path.parent,
            )
            logger.info(f"CV app started with PID {self._process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start CV app: {e}")
            return False

    def wait_for_ready(self) -> bool:
        """Wait for the CV app to be ready.

        Returns:
            True if app is ready, False if timeout
        """
        logger.info(f"Waiting for CV app to be ready at {self.api_url}...")
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            try:
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(f"{self.api_url}/api/status")
                    if response.status_code == 200:
                        logger.info("CV app is ready")
                        return True
            except Exception:
                pass

            time.sleep(0.5)

        logger.error("CV app startup timeout")
        return False

    def stop(self) -> None:
        """Stop the CV Flask app subprocess."""
        if self._process is None:
            return

        logger.info("Stopping CV app...")

        try:
            # Try graceful shutdown first
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                logger.warning("CV app did not terminate gracefully, killing...")
                self._process.kill()
                self._process.wait()

            logger.info("CV app stopped")

        except Exception as e:
            logger.error(f"Error stopping CV app: {e}")

        finally:
            self._process = None


class BeltManager:
    """Manages the conveyor belt."""

    def __init__(self, port: str, baudrate: int = 115200):
        """Initialize the belt manager.

        Args:
            port: Serial port for the belt
            baudrate: Baud rate for serial communication
        """
        self.port = port
        self.baudrate = baudrate
        self._serial: serial.Serial | None = None
        self._belt: BeltControl | None = None
        self._is_running = False

    def connect(self) -> bool:
        """Connect to the belt controller.

        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to belt at {self.port}...")
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self._belt = BeltControl(self._serial)
            logger.info("Belt controller connected")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to belt: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the belt controller."""
        if self._belt:
            self._belt.disable()
        if self._serial:
            self._serial.close()
            self._serial = None
        self._belt = None
        logger.info("Belt controller disconnected")

    def start_belt(self) -> None:
        """Start the belt moving."""
        if self._belt and not self._is_running:
            self._belt.start_belt()
            self._is_running = True

    def stop_belt(self) -> None:
        """Stop the belt."""
        if self._belt and self._is_running:
            self._belt.stop_belt()
            self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if the belt is running."""
        return self._is_running


async def poll_cv_api(api_url: str) -> CVStatus | None:
    """Poll the CV API for status.

    Args:
        api_url: Base URL of the CV API

    Returns:
        CVStatus if successful, None otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{api_url}/api/status")
            if response.status_code == 200:
                return cv_api_status_to_cv_status(response.json())
    except Exception as e:
        logger.warning(f"Failed to poll CV API: {e}")

    return None


async def run_conveyor_computer(args: argparse.Namespace) -> None:
    """Main async function for the conveyor belt computer.

    Args:
        args: Parsed command line arguments
    """
    shutdown_event = asyncio.Event()

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    cv_manager: CVAppManager | None = None
    belt_manager: BeltManager | None = None
    ws_server: ConveyorWebSocketServer | None = None

    try:
        # Start CV app
        cv_manager = CVAppManager(
            app_path=args.cv_app_path,
            api_url=args.cv_api_url,
            startup_timeout=args.cv_startup_timeout,
        )

        if not cv_manager.start():
            logger.error("Failed to start CV app")
            return

        if not cv_manager.wait_for_ready():
            logger.error("CV app did not become ready in time")
            return

        # Start CV detection
        logger.info("Starting CV detection...")
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{args.cv_api_url}/api/start")

        # Connect to belt
        belt_manager = BeltManager(
            port=args.belt_port,
            baudrate=args.belt_baudrate,
        )

        if not belt_manager.connect():
            logger.error("Failed to connect to belt")
            return

        # Track last ACK state for debugging
        last_ack_state: SystemState | None = None
        last_ack_success: bool = True

        async def on_ack(msg: AckMessage) -> None:
            nonlocal last_ack_state, last_ack_success
            last_ack_state = msg.state
            last_ack_success = msg.success

        async def on_error(message: str) -> None:
            logger.error(f"Error from arms computer: {message}")

        # Start WebSocket server
        ws_server = ConveyorWebSocketServer(
            host=args.websocket_host,
            port=args.websocket_port,
            on_ack=on_ack,
            on_error=on_error,
        )
        await ws_server.start()

        # Create state machine
        current_state = SystemState.RUNNING

        async def on_state_change(state: SystemState) -> None:
            nonlocal current_state
            current_state = state

            # Control belt based on state
            if state == SystemState.RUNNING:
                belt_manager.start_belt()
            else:
                belt_manager.stop_belt()

            # Notify arms computer
            if ws_server.is_connected:
                await ws_server.send_state(state)

        state_machine = PackageSortingStateMachine(
            on_state_change=lambda s: asyncio.create_task(on_state_change(s)),
            sorting_to_running_delay=args.transition_delay,
        )

        # Start belt (initial state is RUNNING)
        belt_manager.start_belt()

        logger.info("Conveyor belt computer running. Press Ctrl+C to stop.")
        logger.info(
            f"WebSocket server listening on ws://{args.websocket_host}:{args.websocket_port}"
        )
        logger.info("Waiting for arms computer to connect...")

        # Track for SORTING->RUNNING delay
        sorting_exit_time: float | None = None

        # Main control loop
        while not shutdown_event.is_set():
            # Poll CV API
            cv_status = await poll_cv_api(args.cv_api_url)

            if cv_status is not None:
                old_state = state_machine.state

                # Handle SORTING->RUNNING delay
                if old_state == SystemState.SORTING and not cv_status.package_detected:
                    if sorting_exit_time is None:
                        sorting_exit_time = time.time()
                        logger.info(
                            f"Package disappeared, waiting {args.transition_delay}s before RUNNING"
                        )
                    elif time.time() - sorting_exit_time >= args.transition_delay:
                        # Delay complete, allow transition
                        state_machine.process_cv_status(cv_status)
                        sorting_exit_time = None
                else:
                    sorting_exit_time = None
                    state_machine.process_cv_status(cv_status)

            await asyncio.sleep(args.poll_interval)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Shutting down...")

        if ws_server:
            await ws_server.stop()

        if belt_manager:
            belt_manager.stop_belt()
            belt_manager.disconnect()

        if cv_manager:
            cv_manager.stop()

        logger.info("Conveyor belt computer shutdown complete")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("  CONVEYOR BELT COMPUTER - Package Sorting System")
    logger.info("=" * 50)
    logger.info(f"  Belt port: {args.belt_port}")
    logger.info(f"  CV app: {args.cv_app_path}")
    logger.info(f"  CV API: {args.cv_api_url}")
    logger.info(f"  WebSocket: {args.websocket_host}:{args.websocket_port}")
    logger.info(f"  Transition delay: {args.transition_delay}s")
    logger.info("=" * 50)

    try:
        asyncio.run(run_conveyor_computer(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
