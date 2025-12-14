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
        --sorting-to-running-delay=4.0 \
        --running-to-sorting-delay=8.0 \
        --flipping-to-sorting-delay=4.0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import httpx
import serial
from communication.messages import AckMessage, SystemState
from communication.server import ConveyorWebSocketServer
from state_machine import (
    CVStatus,
    DetectionColor,
    PackageSortingStateMachine,
    cv_api_status_to_cv_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BeltControl:
    def __init__(self, ser: serial.Serial):
        """
        Initialize belt control with a serial connection.
        Immediately disables the belt to prevent unwanted movement.
        """
        self.ser = ser
        self.disable()  # Disable immediately to prevent unwanted movement
        time.sleep(2)  # wait for Arduino reset

    def set_dir(self, sign: int):
        """Set belt direction (1 for forward, -1 for reverse)"""
        self.ser.write(f"DIR {sign}\n".encode())

    def set_speed(self, micros: int):
        """Set belt speed"""
        self.ser.write(f"SPD {micros}\n".encode())

    def enable(self):
        """Enable the belt motor"""
        self.ser.write(b"EN\n")

    def disable(self):
        """Disable the belt motor"""
        self.ser.write(b"DIS\n")

    def start_belt(self):
        """Start the belt moving"""
        self.enable()
        self.set_dir(-1)  # Set direction (1 for forward, -1 for reverse)
        self.set_speed(500)  # Set speed
        print("Belt started")

    def stop_belt(self):
        """Stop the belt"""
        self.disable()
        print("Belt stopped")

    def _read_key(self, timeout: float = 1 / 30):
        """Read a key from terminal stdin (only works when terminal is focused)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # Wait for input with timeout (non-blocking)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if not rlist:
                return None  # No input available
            ch = sys.stdin.read(1)
            # Handle Ctrl+C in raw mode
            if ch == "\x03":
                raise KeyboardInterrupt
            # Handle escape sequences (arrow keys)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "C":
                        return "right"
                    elif ch3 == "D":
                        return "left"
                return "esc"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run(self):
        """Run the keyboard listener for belt control"""
        print(
            "Belt Control - Press RIGHT arrow to start, LEFT arrow to stop, ESC to exit"
        )
        print("Waiting for key presses...")

        try:
            while True:
                key = self._read_key()
                if key is None:
                    continue  # No input, check again at ~30fps
                elif key == "right":
                    self.start_belt()
                elif key == "left":
                    self.stop_belt()
                elif key == "esc":
                    self.stop_belt()
                    break
        except KeyboardInterrupt:
            self.stop_belt()

        # Cleanup
        self.disable()
        self.ser.close()
        print("Exiting...")


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
        "--sorting-to-running-delay",
        type=float,
        default=2.0,
        help="Delay in seconds before transitioning from SORTING to RUNNING after package disappears",
    )
    parser.add_argument(
        "--running-to-sorting-delay",
        type=float,
        default=6.0,
        help="Delay in seconds before transitioning from RUNNING to SORTING/FLIPPING after package is detected",
    )
    parser.add_argument(
        "--flipping-to-sorting-delay",
        type=float,
        default=2.0,
        help="Delay in seconds before transitioning from FLIPPING to SORTING after barcode becomes visible",
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
            # Use uv run from cv_classfication project directory
            cv_project_dir = self.app_path.parent.parent

            # Suppress CV app logs
            self._process = subprocess.Popen(
                ["uv", "run", "python", str(self.app_path)],
                cwd=cv_project_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
            except Exception as e:
                # Don't swallow startup errors; keep this at DEBUG to avoid log spam.
                logger.debug(f"CV app not ready yet: {e}")

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

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Fallback for platforms that don't support add_signal_handler
        logger.warning("Signal handlers not supported on this platform, using fallback")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: shutdown_event.set())

    cv_manager: CVAppManager | None = None
    belt_manager: BeltManager | None = None
    ws_server: ConveyorWebSocketServer | None = None

    try:

        def _create_task_logged(coro: "asyncio.Future", *, name: str) -> asyncio.Task:
            """Create a background task and log any exception it raises.

            This avoids 'swallowed' exceptions from fire-and-forget tasks.
            """
            task = asyncio.create_task(coro, name=name)

            def _done(t: asyncio.Task) -> None:
                try:
                    t.result()
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception(
                        "Unhandled exception in background task %s", t.get_name()
                    )

            task.add_done_callback(_done)
            return task

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
            logger.info(
                "🔔 on_state_change(%s) [ws_connected=%s]",
                state.value,
                ws_server.is_connected,
            )

            # Control belt based on state
            if state == SystemState.RUNNING:
                belt_manager.start_belt()
            else:
                belt_manager.stop_belt()

            # Notify arms computer
            if ws_server.is_connected:
                sent = await ws_server.send_state(state)
                if not sent:
                    logger.warning(
                        "⚠️ Failed to send state %s to arms (ws_connected=%s)",
                        state.value,
                        ws_server.is_connected,
                    )
            else:
                logger.warning(
                    "⚠️ Skipping send_state(%s): no arms client connected",
                    state.value,
                )

        def _schedule_state_change(state: SystemState) -> None:
            _create_task_logged(
                on_state_change(state), name=f"on_state_change({state.value})"
            )

        state_machine = PackageSortingStateMachine(
            on_state_change=_schedule_state_change,
            sorting_to_running_delay=args.sorting_to_running_delay,
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

        # Track for RUNNING->FLIPPING/SORTING delay (when package first detected)
        running_detection_time: float | None = None
        last_package_detected_in_running: bool = False

        # Track for FLIPPING->SORTING delay (when barcode becomes visible)
        flipping_to_sorting_time: float | None = None
        barcode_visible_in_flipping: bool = False

        # Main control loop
        while not shutdown_event.is_set():
            # Poll CV API
            cv_status = await poll_cv_api(args.cv_api_url)

            if shutdown_event.is_set():
                break

            if cv_status is not None:
                old_state = state_machine.state

                # Handle RUNNING->FLIPPING/SORTING delay (when package first detected)
                if old_state == SystemState.RUNNING and cv_status.package_detected:
                    if not last_package_detected_in_running:
                        # Package just detected for the first time in RUNNING state
                        running_detection_time = time.time()
                        last_package_detected_in_running = True
                        logger.info(
                            f"Package detected in RUNNING state, waiting {args.running_to_sorting_delay}s before transitioning"
                        )
                        # Don't process status yet, stay in RUNNING
                    elif running_detection_time is not None:
                        if (
                            time.time() - running_detection_time
                            >= args.running_to_sorting_delay
                        ):
                            # Delay complete, allow transition
                            state_machine.process_cv_status(cv_status)
                            running_detection_time = None
                            # Keep last_package_detected_in_running True since package still detected
                        else:
                            # Still waiting, don't process status yet (stay in RUNNING)
                            pass
                    else:
                        # Already processed transition, continue normally
                        state_machine.process_cv_status(cv_status)
                elif (
                    old_state == SystemState.RUNNING and not cv_status.package_detected
                ):
                    # Package disappeared or not detected, reset tracking
                    running_detection_time = None
                    last_package_detected_in_running = False
                    state_machine.process_cv_status(cv_status)
                # Handle FLIPPING->SORTING delay (when barcode becomes visible)
                elif old_state == SystemState.FLIPPING and cv_status.package_detected:
                    if cv_status.color in (
                        DetectionColor.YELLOW,
                        DetectionColor.RED,
                    ):
                        # Barcode is visible
                        if not barcode_visible_in_flipping:
                            # Barcode just became visible for the first time in FLIPPING state
                            flipping_to_sorting_time = time.time()
                            barcode_visible_in_flipping = True
                            logger.info(
                                f"Barcode visible in FLIPPING state, waiting {args.flipping_to_sorting_delay}s before transitioning to SORTING"
                            )
                            # Don't process status yet, stay in FLIPPING
                        elif flipping_to_sorting_time is not None:
                            if (
                                time.time() - flipping_to_sorting_time
                                >= args.flipping_to_sorting_delay
                            ):
                                # Delay complete, allow transition
                                state_machine.process_cv_status(cv_status)
                                flipping_to_sorting_time = None
                                # Keep barcode_visible_in_flipping True since barcode still visible
                            else:
                                # Still waiting, don't process status yet (stay in FLIPPING)
                                pass
                        else:
                            # Already processed transition, continue normally
                            state_machine.process_cv_status(cv_status)
                    else:
                        # Barcode not visible anymore (shouldn't happen, but reset tracking)
                        flipping_to_sorting_time = None
                        barcode_visible_in_flipping = False
                        state_machine.process_cv_status(cv_status)
                # Handle SORTING->RUNNING delay
                elif (
                    old_state == SystemState.SORTING and not cv_status.package_detected
                ):
                    if sorting_exit_time is None:
                        sorting_exit_time = time.time()
                        logger.info(
                            f"Package disappeared, waiting {args.sorting_to_running_delay}s before RUNNING"
                        )
                    elif (
                        time.time() - sorting_exit_time >= args.sorting_to_running_delay
                    ):
                        # Delay complete, allow transition
                        state_machine.process_cv_status(cv_status)
                        sorting_exit_time = None
                else:
                    sorting_exit_time = None
                    # Reset RUNNING detection tracking when not in RUNNING state
                    running_detection_time = None
                    last_package_detected_in_running = False
                    # Reset FLIPPING->SORTING tracking when not in FLIPPING state
                    flipping_to_sorting_time = None
                    barcode_visible_in_flipping = False
                    state_machine.process_cv_status(cv_status)

            # Sleep in small chunks to check shutdown more frequently
            sleep_chunks = max(1, int(args.poll_interval / 0.01))
            chunk_size = args.poll_interval / sleep_chunks
            for _ in range(sleep_chunks):
                if shutdown_event.is_set():
                    break
                await asyncio.sleep(chunk_size)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Shutting down...")

        # Stop belt first
        if belt_manager:
            try:
                belt_manager.stop_belt()
            except Exception as e:
                logger.error(f"Error stopping belt: {e}")

        # Stop WebSocket server
        if ws_server:
            try:
                await asyncio.wait_for(ws_server.stop(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket server stop timed out")
            except Exception as e:
                logger.error(f"Error stopping WebSocket server: {e}")

        # Disconnect belt
        if belt_manager:
            try:
                belt_manager.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting belt: {e}")

        # Stop CV app last
        if cv_manager:
            try:
                cv_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping CV app: {e}")

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
    logger.info(f"  Sorting->Running delay: {args.sorting_to_running_delay}s")
    logger.info(f"  Running->Sorting delay: {args.running_to_sorting_delay}s")
    logger.info(f"  Flipping->Sorting delay: {args.flipping_to_sorting_delay}s")
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
