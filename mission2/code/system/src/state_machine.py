"""State machine for the package sorting system.

Manages transitions between RUNNING, FLIPPING, and SORTING states
based on CV detection events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from communication.messages import SystemState

logger = logging.getLogger(__name__)


class DetectionColor(str, Enum):
    """Colors detected by the CV system."""

    YELLOW = "YELLOW"
    RED = "RED"
    BLANK = "BLANK"  # Package visible but no tape/barcode visible


@dataclass
class CVStatus:
    """Status from the CV detection system."""

    package_detected: bool
    color: DetectionColor | None  # None if no package or unknown


class PackageSortingStateMachine:
    """State machine for package sorting operations.

    State transitions:
    - RUNNING -> FLIPPING: package detected with BLANK (barcode hidden)
    - RUNNING -> SORTING: package detected with YELLOW/RED (barcode visible)
    - FLIPPING -> SORTING: barcode becomes visible (YELLOW/RED)
    - SORTING -> RUNNING: package disappears from detection zone
    """

    def __init__(
        self,
        on_state_change: Callable[[SystemState], None] | None = None,
        sorting_to_running_delay: float = 2.0,
    ):
        """Initialize the state machine.

        Args:
            on_state_change: Callback when state changes
            sorting_to_running_delay: Delay in seconds before transitioning
                                      from SORTING to RUNNING (to allow robot
                                      to finish placing package)
        """
        self._state = SystemState.RUNNING
        self._on_state_change = on_state_change
        self._sorting_to_running_delay = sorting_to_running_delay

        # Track if we're waiting to transition from SORTING to RUNNING
        self._sorting_complete_pending = False

    @property
    def state(self) -> SystemState:
        """Get the current state."""
        return self._state

    @property
    def sorting_to_running_delay(self) -> float:
        """Get the delay for SORTING -> RUNNING transition."""
        return self._sorting_to_running_delay

    def process_cv_status(self, status: CVStatus) -> SystemState | None:
        """Process CV detection status and potentially transition state.

        Args:
            status: Current CV detection status

        Returns:
            New state if transition occurred, None otherwise
        """
        old_state = self._state
        new_state = self._compute_next_state(status)

        if new_state != old_state:
            self._state = new_state
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")
            if self._on_state_change:
                self._on_state_change(new_state)
            return new_state

        return None

    def _compute_next_state(self, status: CVStatus) -> SystemState:
        """Compute the next state based on current state and CV status.

        Args:
            status: Current CV detection status

        Returns:
            The next state (may be same as current)
        """
        if self._state == SystemState.RUNNING:
            if status.package_detected:
                if status.color == DetectionColor.BLANK:
                    # Package detected but barcode not visible -> need to flip
                    return SystemState.FLIPPING
                elif status.color in (DetectionColor.YELLOW, DetectionColor.RED):
                    # Package detected with visible barcode -> can sort
                    return SystemState.SORTING
            # No package or unknown -> stay in RUNNING
            return SystemState.RUNNING

        elif self._state == SystemState.FLIPPING:
            if status.package_detected:
                if status.color in (DetectionColor.YELLOW, DetectionColor.RED):
                    # Flip complete, barcode now visible -> can sort
                    return SystemState.SORTING
                # Still blank or no detection -> keep flipping
                return SystemState.FLIPPING
            # Package disappeared during flipping (shouldn't happen) -> back to running
            logger.warning("Package disappeared during flipping")
            return SystemState.RUNNING

        elif self._state == SystemState.SORTING:
            if not status.package_detected:
                # Package removed from detection zone -> back to running
                # Note: The actual delay should be handled by the caller
                return SystemState.RUNNING
            # Package still there -> keep sorting
            return SystemState.SORTING

        return self._state

    def force_state(self, state: SystemState) -> None:
        """Force the state machine to a specific state.

        Use with caution - primarily for initialization or error recovery.

        Args:
            state: State to force
        """
        old_state = self._state
        self._state = state
        logger.info(f"State forced: {old_state.value} -> {state.value}")
        if self._on_state_change:
            self._on_state_change(state)


def cv_api_status_to_cv_status(api_response: dict) -> CVStatus:
    """Convert CV Flask API response to CVStatus.

    Args:
        api_response: Response from /api/status endpoint

    Returns:
        CVStatus object
    """
    package_detected = api_response.get("package_detected", False)

    color = None
    color_str = api_response.get("color")
    if color_str:
        try:
            color = DetectionColor(color_str)
        except ValueError:
            logger.warning(f"Unknown color from CV API: {color_str}")

    return CVStatus(package_detected=package_detected, color=color)

