"""WebSocket server for the conveyor belt computer.

This server sends state change commands to the arms computer and receives acknowledgments.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any, Awaitable, Callable

import websockets

from .messages import (
    AckMessage,
    Message,
    PingMessage,
    PongMessage,
    SetStateMessage,
    SystemState,
    parse_message,
    serialize_message,
)

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Get the local network IP address.

    Returns:
        The local IP address as a string, or 'localhost' if unable to determine.
    """
    try:
        # Connect to a remote address to determine the local IP
        # This doesn't actually send data, just determines the route
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # Connect to a non-routable address (doesn't need to be reachable)
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip
    except Exception:
        return "localhost"


class ConveyorWebSocketServer:
    """WebSocket server running on the conveyor belt computer.

    Manages connection to the arms computer and handles message exchange.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_ack: Callable[[AckMessage], Awaitable[None]] | None = None,
        on_error: Callable[[str], Awaitable[None]] | None = None,
    ):
        """Initialize the WebSocket server.

        Args:
            host: Host address to bind to
            port: Port to listen on
            on_ack: Callback when ACK message is received
            on_error: Callback when error message is received
        """
        self.host = host
        self.port = port
        self.on_ack = on_ack
        self.on_error = on_error

        self._server: websockets.WebSocketServer | None = None
        # websockets' server connection type changed in v15; keep this untyped for compatibility.
        self._client: Any | None = None
        self._client_connected = asyncio.Event()
        self._running = False

    @property
    def is_connected(self) -> bool:
        """Check if a client (arms computer) is connected."""
        # Avoid relying on `.open` / `.closed` attributes (changed across websockets versions).
        return self._client is not None

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )
        local_ip = get_local_ip()
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        print(f"\n{'=' * 60}")
        print(f"Server IP address (for client connection): {local_ip}")
        print(f"WebSocket URL: ws://{local_ip}:{self.port}")
        print(f"{'=' * 60}\n")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                # Best-effort shutdown; connection may already be gone.
                pass
            finally:
                self._client = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("WebSocket server stopped")

    async def wait_for_client(self, timeout: float | None = None) -> bool:
        """Wait for a client to connect.

        Args:
            timeout: Maximum time to wait in seconds, None for no timeout

        Returns:
            True if client connected, False if timeout
        """
        try:
            await asyncio.wait_for(self._client_connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def send_state(self, state: SystemState) -> bool:
        """Send a state change command to the arms computer.

        Args:
            state: The new system state

        Returns:
            True if message was sent, False if no client connected
        """
        if self._client is None:
            logger.warning("Cannot send state: no client connected")
            return False

        msg = SetStateMessage(state=state)
        try:
            await self._client.send(serialize_message(msg))
            logger.info(f"Sent state change: {state.value}")
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed while sending state")
            self._client = None
            self._client_connected.clear()
            return False
        except Exception:
            logger.exception("Unexpected error while sending state")
            self._client = None
            self._client_connected.clear()
            return False

    async def ping(self) -> bool:
        """Send a ping to check connection health.

        Returns:
            True if ping was sent, False if no client connected
        """
        if self._client is None:
            return False

        msg = PingMessage()
        try:
            await self._client.send(serialize_message(msg))
            return True
        except websockets.exceptions.ConnectionClosed:
            self._client = None
            self._client_connected.clear()
            return False
        except Exception:
            logger.exception("Unexpected error while sending ping")
            self._client = None
            self._client_connected.clear()
            return False

    async def _handle_client(self, websocket: Any) -> None:
        """Handle a client connection."""
        # Only allow one client at a time
        if self._client is not None:
            logger.warning("Rejecting new client: already connected")
            await websocket.close(1008, "Only one client allowed")
            return

        self._client = websocket
        self._client_connected.set()
        logger.info(f"Arms computer connected from {websocket.remote_address}")

        try:
            async for raw_message in websocket:
                await self._handle_message(raw_message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client disconnected: {e}")
        finally:
            self._client = None
            self._client_connected.clear()
            logger.info("Arms computer disconnected")

    async def _handle_message(self, raw_message: str | bytes) -> None:
        """Handle an incoming message from the arms computer."""
        try:
            msg = parse_message(raw_message)
        except ValueError as e:
            logger.error(f"Failed to parse message: {e}")
            return

        if isinstance(msg, AckMessage):
            logger.info(
                f"Received ACK for state {msg.state.value}: success={msg.success}"
            )
            if self.on_ack:
                await self.on_ack(msg)

        elif isinstance(msg, PongMessage):
            logger.debug("Received PONG")

        elif hasattr(msg, "type"):
            from .messages import MessageType

            if msg.type == MessageType.ERROR:
                logger.error(f"Received error from arms: {msg.message}")
                if self.on_error:
                    await self.on_error(msg.message)
            else:
                logger.warning(f"Unexpected message type from arms: {msg.type}")
