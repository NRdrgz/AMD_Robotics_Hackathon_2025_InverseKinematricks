"""Message definitions for WebSocket communication between arms and conveyor computers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Union


class MessageType(str, Enum):
    """Types of messages exchanged between computers."""

    SET_STATE = "set_state"
    ACK = "ack"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


class SystemState(str, Enum):
    """System states for the package sorting operation."""

    STOPPED = "STOPPED"  # Disconnected from server, no policy inference or actions
    RUNNING = "RUNNING"  # Belt on, blue pick running, black idle
    FLIPPING = "FLIPPING"  # Belt off, black flip running
    SORTING = "SORTING"  # Belt off, black sort running


@dataclass
class SetStateMessage:
    """Command from conveyor to arms to change system state."""

    state: SystemState

    @property
    def type(self) -> MessageType:
        return MessageType.SET_STATE

    def to_dict(self) -> dict:
        return {"type": self.type.value, "state": self.state.value}


@dataclass
class AckMessage:
    """Acknowledgment from arms to conveyor after state change."""

    state: SystemState
    success: bool

    @property
    def type(self) -> MessageType:
        return MessageType.ACK

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "state": self.state.value,
            "success": self.success,
        }


@dataclass
class ErrorMessage:
    """Error message from arms to conveyor."""

    message: str

    @property
    def type(self) -> MessageType:
        return MessageType.ERROR

    def to_dict(self) -> dict:
        return {"type": self.type.value, "message": self.message}


@dataclass
class PingMessage:
    """Ping message for connection health check."""

    @property
    def type(self) -> MessageType:
        return MessageType.PING

    def to_dict(self) -> dict:
        return {"type": self.type.value}


@dataclass
class PongMessage:
    """Pong response to ping."""

    @property
    def type(self) -> MessageType:
        return MessageType.PONG

    def to_dict(self) -> dict:
        return {"type": self.type.value}


# Union type for all message types
Message = Union[SetStateMessage, AckMessage, ErrorMessage, PingMessage, PongMessage]


def parse_message(data: str | bytes) -> Message:
    """Parse a JSON message string into the appropriate message type.

    Args:
        data: JSON string or bytes containing the message

    Returns:
        Parsed message object

    Raises:
        ValueError: If the message type is unknown or data is invalid
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")

    try:
        msg_dict = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    msg_type = msg_dict.get("type")
    if not msg_type:
        raise ValueError("Message missing 'type' field")

    try:
        msg_type_enum = MessageType(msg_type)
    except ValueError:
        raise ValueError(f"Unknown message type: {msg_type}")

    if msg_type_enum == MessageType.SET_STATE:
        state_str = msg_dict.get("state")
        if not state_str:
            raise ValueError("SET_STATE message missing 'state' field")
        try:
            state = SystemState(state_str)
        except ValueError:
            raise ValueError(f"Unknown state: {state_str}")
        return SetStateMessage(state=state)

    elif msg_type_enum == MessageType.ACK:
        state_str = msg_dict.get("state")
        success = msg_dict.get("success")
        if state_str is None or success is None:
            raise ValueError("ACK message missing 'state' or 'success' field")
        try:
            state = SystemState(state_str)
        except ValueError:
            raise ValueError(f"Unknown state: {state_str}")
        return AckMessage(state=state, success=success)

    elif msg_type_enum == MessageType.ERROR:
        message = msg_dict.get("message", "")
        return ErrorMessage(message=message)

    elif msg_type_enum == MessageType.PING:
        return PingMessage()

    elif msg_type_enum == MessageType.PONG:
        return PongMessage()

    else:
        raise ValueError(f"Unhandled message type: {msg_type}")


def serialize_message(msg: Message) -> str:
    """Serialize a message object to JSON string.

    Args:
        msg: Message object to serialize

    Returns:
        JSON string representation of the message
    """
    return json.dumps(msg.to_dict())
