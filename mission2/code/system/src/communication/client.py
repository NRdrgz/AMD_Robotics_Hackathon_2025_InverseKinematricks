"""WebSocket client for the arms computer.

This client receives state change commands from the conveyor belt computer
and sends back acknowledgments.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import websockets
from websockets.client import WebSocketClientProtocol

from .messages import (
    AckMessage,
    ErrorMessage,
    Message,
    PingMessage,
    PongMessage,
    SetStateMessage,
    SystemState,
    parse_message,
    serialize_message,
)

logger = logging.getLogger(__name__)


class ArmsWebSocketClient:
    """WebSocket client running on the arms computer.

    Connects to the conveyor belt computer and handles state change commands.
    """

    def __init__(
        self,
        host: str,
        port: int = 8765,
        on_state_change: Callable[[SystemState], Awaitable[bool]] | None = None,
        on_connected: Callable[[], Awaitable[None]] | None = None,
        on_disconnected: Callable[[], Awaitable[None]] | None = None,
        reconnect_interval: float = 2.0,
    ):
        """Initialize the WebSocket client.

        Args:
            host: Host address of the conveyor computer
            port: Port to connect to
            on_state_change: Callback when state change is received.
                             Should return True if state change was successful.
            on_connected: Callback when connection is established
            on_disconnected: Callback when connection is lost
            reconnect_interval: Seconds to wait before reconnecting after disconnect
        """
        self.host = host
        self.port = port
        self.on_state_change = on_state_change
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.reconnect_interval = reconnect_interval

        self._websocket: WebSocketClientProtocol | None = None
        self._running = False
        self._connected = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        ws = self._websocket
        if ws is None:
            return False
        # Compatibility across websockets versions (the connection object API changed in v15).
        try:
            open_attr = getattr(ws, "open", None)
            if isinstance(open_attr, bool):
                return open_attr
            closed_attr = getattr(ws, "closed", None)
            if isinstance(closed_attr, bool):
                return not closed_attr
        except Exception:
            # Fall through to a conservative best-effort.
            pass
        return True

    @property
    def uri(self) -> str:
        """Get the WebSocket URI."""
        return f"ws://{self.host}:{self.port}"

    async def start(self) -> None:
        """Start the client and begin connecting."""
        self._running = True
        self._task = asyncio.create_task(self._connection_loop())
        logger.info(f"WebSocket client starting, will connect to {self.uri}")

    async def stop(self) -> None:
        """Stop the client and disconnect."""
        self._running = False
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._connected.clear()
        logger.info("WebSocket client stopped")

    async def wait_for_connection(self, timeout: float | None = None) -> bool:
        """Wait for connection to be established.

        Args:
            timeout: Maximum time to wait in seconds, None for no timeout

        Returns:
            True if connected, False if timeout
        """
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def send_ack(self, state: SystemState, success: bool) -> bool:
        """Send an acknowledgment message.

        Args:
            state: The state being acknowledged
            success: Whether the state change was successful

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cannot send ACK: not connected")
            return False

        msg = AckMessage(state=state, success=success)
        try:
            await self._websocket.send(serialize_message(msg))
            logger.debug(f"Sent ACK for state {state.value}: success={success}")
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed while sending ACK")
            return False

    async def send_error(self, message: str) -> bool:
        """Send an error message.

        Args:
            message: Error description

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cannot send error: not connected")
            return False

        msg = ErrorMessage(message=message)
        try:
            await self._websocket.send(serialize_message(msg))
            logger.debug(f"Sent error: {message}")
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed while sending error")
            return False

    async def _connection_loop(self) -> None:
        """Main connection loop with automatic reconnection."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")

            if self._running:
                logger.info(f"Reconnecting in {self.reconnect_interval}s...")
                await asyncio.sleep(self.reconnect_interval)

    async def _connect_and_listen(self) -> None:
        """Connect to the server and listen for messages."""
        try:
            async with websockets.connect(self.uri) as websocket:
                self._websocket = websocket
                self._connected.set()
                logger.info(f"Connected to conveyor computer at {self.uri}")

                # Notify connection established
                if self.on_connected:
                    try:
                        await self.on_connected()
                    except Exception as e:
                        logger.error(f"Error in on_connected callback: {e}")

                try:
                    async for raw_message in websocket:
                        await self._handle_message(raw_message)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"Disconnected from server: {e}")
        finally:
            self._websocket = None
            self._connected.clear()

            # Notify connection lost
            if self.on_disconnected:
                try:
                    await self.on_disconnected()
                except Exception as e:
                    logger.error(f"Error in on_disconnected callback: {e}")

    async def _handle_message(self, raw_message: str | bytes) -> None:
        """Handle an incoming message from the conveyor computer."""
        try:
            msg = parse_message(raw_message)
        except ValueError as e:
            logger.error(f"Failed to parse message: {e}")
            return

        if isinstance(msg, SetStateMessage):
            logger.info(f"Received state change command: {msg.state.value}")
            success = True
            if self.on_state_change:
                try:
                    success = await self.on_state_change(msg.state)
                except Exception as e:
                    logger.error(f"Error handling state change: {e}")
                    success = False
                    await self.send_error(str(e))

            # Always send ACK after handling state change
            await self.send_ack(msg.state, success)

        elif isinstance(msg, PingMessage):
            logger.debug("Received PING, sending PONG")
            pong = PongMessage()
            try:
                await self._websocket.send(serialize_message(pong))
            except websockets.exceptions.ConnectionClosed:
                pass

        else:
            logger.warning(f"Unexpected message type: {type(msg)}")
