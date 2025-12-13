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
    make_default_robot_observation_processor,
)
from lerobot.robots import Robot
from lerobot.robots.utils import make_robot_from_config
from policies.base import PolicyWrapper
from torch import Tensor

if TYPE_CHECKING:
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
    ):
        """Initialize the action executor.

        Args:
            arm_id: Identifier for this arm
            robot: Robot wrapper instance
            shutdown_event: Event to signal shutdown
            fps: Action execution frequency
        """
        self.arm_id = arm_id
        self.robot = robot
        self.shutdown_event = shutdown_event
        self.fps = fps

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
            self._action_queue.put(None)
            self._thread.join(timeout=2.0)
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
                start_time = time.perf_counter()

                try:
                    action = self._action_queue.get(timeout=0.1)
                except Empty:
                    continue

                if action is None:
                    # Shutdown signal
                    break

                # Process and send action
                action_cpu = action.cpu()
                action_dict = {
                    key: action_cpu[i].item()
                    for i, key in enumerate(self.robot.action_features)
                }
                action_processed = self._robot_action_processor((action_dict, None))
                self.robot.send_action(action_processed)
                action_count += 1

                # Maintain timing using time.sleep
                dt_s = time.perf_counter() - start_time
                busy_wait_time = action_interval - dt_s
                if busy_wait_time > 0:
                    time.sleep(busy_wait_time)

        except Exception as e:
            logger.error(f"Error in {self.arm_id.value} executor: {e}")
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
    ):
        """Initialize the policy runner.

        Args:
            arm_id: Identifier for this arm
            robot: Robot wrapper instance
            executor: Action executor for this arm
            shutdown_event: Event to signal shutdown
            fps: Inference frequency
        """
        self.arm_id = arm_id
        self.robot = robot
        self.executor = executor
        self.shutdown_event = shutdown_event
        self.fps = fps

        self._current_policy: PolicyWrapper | None = None
        self._policy_lock = Lock()
        self._thread: Thread | None = None
        self._active = Event()  # Whether this runner should be producing actions
        self._robot_observation_processor = make_default_robot_observation_processor()

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
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_policy(self, policy: PolicyWrapper | None) -> None:
        """Set the active policy for this arm.

        Args:
            policy: Policy to run, or None to idle
        """
        with self._policy_lock:
            if self._current_policy != policy:
                # Clear pending actions when switching policies
                self.executor.clear_queue()
                self._current_policy = policy
                if policy:
                    policy.reset()
                    logger.info(
                        f"{self.arm_id.value} arm: policy set to {policy.config.hf_path}"
                    )
                else:
                    logger.info(f"{self.arm_id.value} arm: policy cleared (idle)")

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
                # Wait until active
                if not self._active.wait(timeout=0.1):
                    continue

                start_time = time.perf_counter()

                with self._policy_lock:
                    policy = self._current_policy

                if policy is None:
                    time.sleep(0.1)
                    continue

                try:
                    # Get observation
                    obs = self.robot.get_observation()
                    obs_processed = self._robot_observation_processor(obs)

                    # Get action from policy
                    action = policy.get_action(
                        obs_processed,
                        self.robot.observation_features,
                    )

                    if action is not None:
                        self.executor.put_action(action)

                except Exception as e:
                    logger.error(f"Error in {self.arm_id.value} inference: {e}")

                # Maintain timing using time.sleep
                dt_s = time.perf_counter() - start_time
                busy_wait_time = inference_interval - dt_s
                if busy_wait_time > 0:
                    time.sleep(busy_wait_time)

        except Exception as e:
            logger.error(f"Fatal error in {self.arm_id.value} policy runner: {e}")
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
    ):
        """Initialize the arm controller.

        Args:
            blue_arm_config: Configuration for the blue arm
            black_arm_config: Configuration for the black arm
            pick_policy: Policy for picking (blue arm)
            flip_policy: Policy for flipping (black arm)
            sort_policy: Policy for sorting (black arm)
            fps: Action/inference frequency
        """
        self.fps = fps
        self._shutdown_event = Event()
        self._state_lock = Lock()
        self._current_state = SystemState.RUNNING

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
            ArmId.BLUE, self._blue_robot, self._shutdown_event, fps
        )
        self._black_executor = ActionExecutor(
            ArmId.BLACK, self._black_robot, self._shutdown_event, fps
        )

        # Create policy runners
        self._blue_runner = PolicyRunner(
            ArmId.BLUE, self._blue_robot, self._blue_executor, self._shutdown_event, fps
        )
        self._black_runner = PolicyRunner(
            ArmId.BLACK,
            self._black_robot,
            self._black_executor,
            self._shutdown_event,
            fps,
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
        self._apply_state(self._current_state)

        logger.info("Arm controller started")

    def stop(self) -> None:
        """Stop all threads and cleanup."""
        logger.info("Stopping arm controller...")
        self._shutdown_event.set()

        # Stop runners first
        self._blue_runner.stop()
        self._black_runner.stop()

        # Then stop executors
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

        logger.info(f"State change: {old_state.value} -> {state.value}")
        self._apply_state(state)
        return True

    def get_state(self) -> SystemState:
        """Get the current system state."""
        with self._state_lock:
            return self._current_state

    def _apply_state(self, state: SystemState) -> None:
        """Apply the given state to the arm controllers.

        State table:
        - RUNNING: blue=pick (active), black=idle
        - FLIPPING: blue=idle, black=flip (active)
        - SORTING: blue=idle, black=sort (active)
        """
        if state == SystemState.RUNNING:
            # Blue arm runs pick policy
            self._blue_runner.set_policy(self.pick_policy)
            self._blue_runner.activate()

            # Black arm is idle
            self._black_runner.deactivate()
            self._black_runner.set_policy(None)

        elif state == SystemState.FLIPPING:
            # Blue arm is idle
            self._blue_runner.deactivate()
            self._blue_runner.set_policy(None)

            # Black arm runs flip policy
            self._black_runner.set_policy(self.flip_policy)
            self._black_runner.activate()

        elif state == SystemState.SORTING:
            # Blue arm is idle
            self._blue_runner.deactivate()
            self._blue_runner.set_policy(None)

            # Black arm runs sort policy
            self._black_runner.set_policy(self.sort_policy)
            self._black_runner.activate()

        else:
            logger.warning(f"Unknown state: {state}")
