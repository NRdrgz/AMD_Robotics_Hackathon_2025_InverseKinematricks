"""Arm controller for managing robots, policies, and action execution.

This module coordinates the two robot arms (blue and black) with their
respective policies based on the current system state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

from communication.messages import SystemState
from lerobot.processor.factory import (
    make_default_robot_action_processor,
)
from lerobot.robots import Robot
from lerobot.robots.utils import make_robot_from_config
from policies.base import PolicyWrapper
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from lerobot.robots import RobotConfig

logger = logging.getLogger(__name__)


class ArmId(str, Enum):
    """Identifier for robot arms."""

    BLUE = "blue"
    BLACK = "black"


@dataclass
class ArmConfig:
    """Configuration for a robot arm."""

    arm_id: ArmId
    robot_config: "RobotConfig"


class RobotWrapper:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self._lock = Lock()
        # Cache features after connection (they don't change)
        self._observation_features: list[str] | None = None
        self._action_features: list[str] | None = None

    def get_observation(self) -> dict[str, Tensor]:
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor | dict) -> None:
        with self._lock:
            self.robot.send_action(action)

    @property
    def observation_features(self) -> list[str]:
        if self._observation_features is None:
            with self._lock:
                self._observation_features = self.robot.observation_features
        return self._observation_features

    @property
    def action_features(self) -> list[str]:
        if self._action_features is None:
            with self._lock:
                self._action_features = self.robot.action_features
        return self._action_features

    def connect(self) -> None:
        with self._lock:
            self.robot.connect()
        # Cache features after connection
        self._observation_features = self.robot.observation_features
        self._action_features = self.robot.action_features

    def disconnect(self) -> None:
        with self._lock:
            self.robot.disconnect()
        self._observation_features = None
        self._action_features = None


class ActionExecutor:
    """Executes actions on a robot arm from an action queue."""

    def __init__(
        self,
        arm_id: ArmId,
        robot: RobotWrapper,
        shutdown_event: Event,
        fps: float = 30.0,
        connection_check: "Callable[[], bool] | None" = None,
    ):
        """Initialize the action executor.

        Args:
            arm_id: Identifier for this arm
            robot: Robot wrapper instance
            shutdown_event: Event to signal shutdown
            fps: Action execution frequency
            connection_check: Optional callable that returns True if connected to server
        """
        self.arm_id = arm_id
        self.robot = robot
        self.shutdown_event = shutdown_event
        self.fps = fps
        self._connection_check = connection_check

        self._action_queue: Queue[Tensor | None] = Queue()
        self._thread: Thread | None = None
        self._robot_action_processor = make_default_robot_action_processor()

    def start(self) -> None:
        """Start the action executor thread."""
        self._thread = Thread(
            target=self._run,
            daemon=True,
            name=f"{self.arm_id.value.capitalize()}ActorThread",
        )
        self._thread.start()
        logger.info(f"Started {self.arm_id.value} arm executor thread")

    def stop(self) -> None:
        """Stop the action executor thread."""
        if self._thread and self._thread.is_alive():
            # Put None to unblock the queue
            try:
                self._action_queue.put(None)
            except Exception as e:
                # Don't swallow exceptions; this can hide shutdown issues.
                logger.debug(
                    "Failed to unblock action queue during stop(): %s", e, exc_info=True
                )

            # Wait briefly for thread to stop - force exit timer will handle stuck threads
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning(
                    f"{self.arm_id.value.capitalize()} executor thread did not stop in 1s"
                )
            self._thread = None

    def put_action(self, action: Tensor) -> None:
        """Add an action to the queue.

        Args:
            action: Action tensor to execute
        """
        self._action_queue.put(action)

    def clear_queue(self) -> None:
        """Clear all pending actions from the queue."""
        while not self._action_queue.empty():
            try:
                self._action_queue.get_nowait()
            except Empty:
                break

    @property
    def queue_size(self) -> int:
        """Get the current queue size."""
        return self._action_queue.qsize()

    def _run(self) -> None:
        """Main executor loop."""
        action_interval = 1.0 / self.fps
        action_count = 0

        try:
            while not self.shutdown_event.is_set():
                # Check server connection before processing actions
                if self._connection_check is not None and not self._connection_check():
                    # Not connected to server, skip action execution
                    # Clear any pending actions since we can't execute them safely
                    while not self._action_queue.empty():
                        try:
                            self._action_queue.get_nowait()
                        except Empty:
                            break
                    time.sleep(0.1)
                    continue

                start_time = time.perf_counter()

                try:
                    action = self._action_queue.get(timeout=0.1)
                except Empty:
                    # Check shutdown while waiting
                    if self.shutdown_event.is_set():
                        break
                    continue

                if action is None:
                    # Shutdown signal
                    break

                if self.shutdown_event.is_set():
                    break

                # Double-check connection before sending action
                if self._connection_check is not None and not self._connection_check():
                    # Lost connection, skip this action
                    continue

                # Process and send action
                try:
                    action_cpu = action.cpu()
                    action_dict = {
                        key: action_cpu[i].item()
                        for i, key in enumerate(self.robot.action_features)
                    }
                    action_processed = self._robot_action_processor((action_dict, None))

                    if self.shutdown_event.is_set():
                        break

                    self.robot.send_action(action_processed)
                    action_count += 1
                except Exception as e:
                    logger.exception(
                        f"Error processing action in {self.arm_id.value} executor: {e}"
                    )
                    # Continue loop even on error, but check shutdown
                    if self.shutdown_event.is_set():
                        break
                    continue

                if self.shutdown_event.is_set():
                    break

                # Maintain timing using time.sleep, checking shutdown frequently
                dt_s = time.perf_counter() - start_time
                busy_wait_time = action_interval - dt_s
                if busy_wait_time > 0:
                    # Sleep in small chunks to check shutdown frequently
                    sleep_chunks = max(1, int(busy_wait_time / 0.01))
                    chunk_size = busy_wait_time / sleep_chunks
                    for _ in range(sleep_chunks):
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(chunk_size)

        except Exception as e:
            logger.exception(f"Error in {self.arm_id.value} executor: {e}")
            raise

        logger.info(
            f"{self.arm_id.value.capitalize()} executor stopped. Actions executed: {action_count}"
        )


class PolicyRunner:
    """Runs policy inference and feeds actions to the executor."""

    def __init__(
        self,
        arm_id: ArmId,
        robot: RobotWrapper,
        executor: ActionExecutor,
        shutdown_event: Event,
        fps: float = 30.0,
        queue_threshold: int = 0,
        connection_check: "Callable[[], bool] | None" = None,
    ):
        """Initialize the policy runner.

        Args:
            arm_id: Identifier for this arm
            robot: Robot wrapper instance
            executor: Action executor for this arm
            shutdown_event: Event to signal shutdown
            fps: Inference frequency
            queue_threshold: Only run inference when executor queue size <= this value.
                             Set to 0 to only infer when queue is empty (recommended
                             for ACT policies without RTC).
            connection_check: Optional callable that returns True if connected to server
        """
        self.arm_id = arm_id
        self.robot = robot
        self.executor = executor
        self.shutdown_event = shutdown_event
        self.fps = fps
        self.queue_threshold = queue_threshold
        self._connection_check = connection_check

        self._current_policy: PolicyWrapper | None = None
        self._policy_lock = Lock()
        self._thread: Thread | None = None
        self._active = Event()  # Whether this runner should be producing actions

    def start(self) -> None:
        """Start the policy runner thread."""
        self._thread = Thread(
            target=self._run,
            daemon=True,
            name=f"{self.arm_id.value.capitalize()}InferenceThread",
        )
        self._thread.start()
        logger.info(f"Started {self.arm_id.value} arm policy runner thread")

    def stop(self) -> None:
        """Stop the policy runner thread."""
        self._active.clear()
        if self._thread and self._thread.is_alive():
            # Wait briefly for thread to stop - force exit timer will handle stuck threads
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning(
                    f"{self.arm_id.value.capitalize()} policy runner thread did not stop in 1s"
                )
            self._thread = None

    def set_policy(self, policy: PolicyWrapper | None) -> None:
        """Set the active policy for this arm.

        Args:
            policy: Policy to run, or None to idle
        """
        with self._policy_lock:
            old_policy_name = (
                self._current_policy.config.hf_path
                if self._current_policy
                else "None (idle)"
            )

            if self._current_policy != policy:
                # Clear pending actions when switching policies
                queue_size = self.executor.queue_size
                if queue_size > 0:
                    logger.info(
                        f"  🧹 {self.arm_id.value.capitalize()} arm: Clearing {queue_size} pending actions"
                    )
                self.executor.clear_queue()

                self._current_policy = policy
                if policy:
                    policy.reset()
                    new_policy_name = policy.config.hf_path
                    logger.info(
                        f"  🤖 POLICY CHANGE [{self.arm_id.value.upper()} ARM]: "
                        f"{old_policy_name} → {new_policy_name}"
                    )
                else:
                    logger.info(
                        f"  🤖 POLICY CHANGE [{self.arm_id.value.upper()} ARM]: "
                        f"{old_policy_name} → None (idle)"
                    )

    def activate(self) -> None:
        """Activate the policy runner to start producing actions."""
        self._active.set()

    def deactivate(self) -> None:
        """Deactivate the policy runner (stop producing actions)."""
        self._active.clear()
        self.executor.clear_queue()

    def _run(self) -> None:
        """Main inference loop."""
        inference_interval = 1.0 / self.fps

        try:
            while not self.shutdown_event.is_set():
                # Check server connection before running inference
                if self._connection_check is not None and not self._connection_check():
                    # Not connected to server, skip inference
                    time.sleep(0.1)
                    continue

                # Wait until active, checking shutdown frequently
                if not self._active.wait(timeout=0.1):
                    if self.shutdown_event.is_set():
                        break
                    continue

                if self.shutdown_event.is_set():
                    break

                # Double-check connection after wait
                if self._connection_check is not None and not self._connection_check():
                    # Lost connection while waiting
                    continue

                with self._policy_lock:
                    policy = self._current_policy

                if policy is None:
                    # Sleep in small chunks to check shutdown frequently
                    for _ in range(10):  # 10 * 0.01s = 0.1s total
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(0.01)
                    continue

                if self.shutdown_event.is_set():
                    break

                # Only run inference when queue is low enough
                # This matches the pattern in eval_with_real_robot.py
                if self.executor.queue_size > self.queue_threshold:
                    # Queue has enough actions, sleep briefly to avoid busy waiting
                    # Check shutdown during sleep
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(0.01)
                    continue

                if self.shutdown_event.is_set():
                    break

                start_time = time.perf_counter()

                try:
                    # Get observation - this might block, but we check shutdown before
                    if self.shutdown_event.is_set():
                        break
                    obs = self.robot.get_observation()

                    if self.shutdown_event.is_set():
                        break

                    # Get action from policy - this might block, but we check shutdown before
                    action = policy.get_action(
                        obs,
                        self.robot.observation_features,
                    )

                    if self.shutdown_event.is_set():
                        break

                    if action is not None:
                        self.executor.put_action(action)

                except Exception as e:
                    logger.exception(f"Error in {self.arm_id.value} inference: {e}")
                    # Continue loop even on error, but check shutdown
                    if self.shutdown_event.is_set():
                        break

                if self.shutdown_event.is_set():
                    break

                # Maintain timing using time.sleep, checking shutdown frequently
                dt_s = time.perf_counter() - start_time
                busy_wait_time = inference_interval - dt_s
                if busy_wait_time > 0:
                    # Sleep in small chunks to check shutdown frequently
                    sleep_chunks = max(1, int(busy_wait_time / 0.01))
                    chunk_size = busy_wait_time / sleep_chunks
                    for _ in range(sleep_chunks):
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(chunk_size)

        except Exception as e:
            logger.exception(f"Fatal error in {self.arm_id.value} policy runner: {e}")
            raise

        logger.info(f"{self.arm_id.value.capitalize()} policy runner stopped")


class ArmController:
    """Main controller coordinating both robot arms and their policies."""

    def __init__(
        self,
        blue_arm_config: "RobotConfig",
        black_arm_config: "RobotConfig",
        pick_policy: PolicyWrapper,
        flip_policy: PolicyWrapper,
        sort_policy: PolicyWrapper,
        fps: float = 30.0,
        shutdown_event: Event | None = None,
        connection_check: "Callable[[], bool] | None" = None,
        queue_threshold: int = 0,
    ):
        """Initialize the arm controller.

        Args:
            blue_arm_config: Configuration for the blue arm
            black_arm_config: Configuration for the black arm
            pick_policy: Policy for picking (blue arm)
            flip_policy: Policy for flipping (black arm)
            sort_policy: Policy for sorting (black arm)
            fps: Action/inference frequency
            shutdown_event: Optional shutdown event to use. If None, creates a new one.
            connection_check: Optional callable that returns True if connected to server.
                              When provided, policy inference and actions are only executed
                              while connected.
            queue_threshold: Only run inference when executor queue size <= this value.
                             Set to 0 to only infer when queue is empty (recommended
                             for ACT policies without RTC).
        """
        self.fps = fps
        self._shutdown_event = shutdown_event if shutdown_event is not None else Event()
        self._connection_check = connection_check
        self._state_lock = Lock()
        self._current_state = SystemState.STOPPED

        # Store policies
        self.pick_policy = pick_policy
        self.flip_policy = flip_policy
        self.sort_policy = sort_policy

        # Create robots
        logger.info("Creating robot instances...")
        self._blue_robot = RobotWrapper(make_robot_from_config(blue_arm_config))
        self._black_robot = RobotWrapper(make_robot_from_config(black_arm_config))

        # Create executors
        self._blue_executor = ActionExecutor(
            ArmId.BLUE, self._blue_robot, self._shutdown_event, fps, connection_check
        )
        self._black_executor = ActionExecutor(
            ArmId.BLACK, self._black_robot, self._shutdown_event, fps, connection_check
        )

        # Create policy runners
        self._blue_runner = PolicyRunner(
            ArmId.BLUE,
            self._blue_robot,
            self._blue_executor,
            self._shutdown_event,
            fps,
            queue_threshold=queue_threshold,
            connection_check=connection_check,
        )
        self._black_runner = PolicyRunner(
            ArmId.BLACK,
            self._black_robot,
            self._black_executor,
            self._shutdown_event,
            fps,
            queue_threshold=queue_threshold,
            connection_check=connection_check,
        )

    def connect(self) -> None:
        """Connect to both robots."""
        logger.info("Connecting to robots...")
        self._blue_robot.connect()
        logger.info("Blue arm connected")
        self._black_robot.connect()
        logger.info("Black arm connected")

    def disconnect(self) -> None:
        """Disconnect from both robots."""
        logger.info("Disconnecting robots...")
        self._blue_robot.disconnect()
        self._black_robot.disconnect()
        logger.info("Robots disconnected")

    def start(self) -> None:
        """Start all executor and runner threads."""
        logger.info("Starting arm controller...")

        # Start executors
        self._blue_executor.start()
        self._black_executor.start()

        # Start runners
        self._blue_runner.start()
        self._black_runner.start()

        # Apply initial state
        logger.info(f"Applying initial state: {self._current_state.value}")
        self._apply_state(self._current_state)

        logger.info("Arm controller started")

    def stop(self) -> None:
        """Stop all threads and cleanup."""
        logger.info("Stopping arm controller...")

        # Set shutdown event first to signal all threads
        self._shutdown_event.set()

        # Deactivate runners to stop producing new actions
        self._blue_runner.deactivate()
        self._black_runner.deactivate()

        # Stop runners first (they feed executors)
        logger.info("Stopping policy runners...")
        self._blue_runner.stop()
        self._black_runner.stop()

        # Then stop executors
        logger.info("Stopping action executors...")
        self._blue_executor.stop()
        self._black_executor.stop()

        logger.info("Arm controller stopped")

    def set_state(self, state: SystemState) -> bool:
        """Set the system state and configure arms accordingly.

        Args:
            state: New system state

        Returns:
            True if state change was successful
        """
        with self._state_lock:
            old_state = self._current_state
            self._current_state = state

        logger.info(f"🔄 STATE CHANGE: {old_state.value} -> {state.value}")
        self._apply_state(state)
        return True

    def get_state(self) -> SystemState:
        """Get the current system state."""
        with self._state_lock:
            return self._current_state

    def _apply_state(self, state: SystemState) -> None:
        """Apply the given state to the arm controllers.

        State table:
        - STOPPED: blue=idle, black=idle (disconnected from server)
        - RUNNING: blue=pick (active), black=idle
        - FLIPPING: blue=idle, black=flip (active)
        - SORTING: blue=idle, black=sort (active)
        """
        logger.info(f"📋 Applying state configuration for {state.value}...")

        if state == SystemState.STOPPED:
            # Both arms are idle - disconnected from server
            logger.info("  → Blue arm: Clearing policy and deactivating (STOPPED)")
            self._blue_runner.deactivate()
            self._blue_runner.set_policy(None)

            logger.info("  → Black arm: Clearing policy and deactivating (STOPPED)")
            self._black_runner.deactivate()
            self._black_runner.set_policy(None)

        elif state == SystemState.RUNNING:
            # Blue arm runs pick policy
            logger.info("  → Blue arm: Setting PICK policy and activating")
            self._blue_runner.set_policy(self.pick_policy)
            self._blue_runner.activate()

            # Black arm is idle
            logger.info("  → Black arm: Clearing policy and deactivating")
            self._black_runner.deactivate()
            self._black_runner.set_policy(None)

        elif state == SystemState.FLIPPING:
            # Blue arm is idle
            logger.info("  → Blue arm: Clearing policy and deactivating")
            self._blue_runner.deactivate()
            self._blue_runner.set_policy(None)

            # Black arm runs flip policy
            logger.info("  → Black arm: Setting FLIP policy and activating")
            self._black_runner.set_policy(self.flip_policy)
            self._black_runner.activate()

        elif state == SystemState.SORTING:
            # Blue arm is idle
            logger.info("  → Blue arm: Clearing policy and deactivating")
            self._blue_runner.deactivate()
            self._blue_runner.set_policy(None)

            # Black arm runs sort policy
            logger.info("  → Black arm: Setting SORT policy and activating")
            self._black_runner.set_policy(self.sort_policy)
            self._black_runner.activate()

        else:
            logger.warning(f"⚠️  Unknown state: {state}")

        logger.info(f"✅ State configuration for {state.value} applied successfully")
