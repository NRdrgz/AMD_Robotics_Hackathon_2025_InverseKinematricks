#!/usr/bin/env python
"""Fake conveyor server for debugging WebSocket connection.

This script creates a simple WebSocket server that:
- Accepts connections from arms computers
- Sends state change commands when you press keys
- Logs all messages received

Usage:
    uv run python src/fake_conveyor_server.py

Keys:
    r - Send RUNNING state
    f - Send FLIPPING state
    s - Send SORTING state
    x - Send STOPPED state
    p - Send PING
    q - Quit
"""

from __future__ import annotations

import asyncio
import logging
import signal

import websockets
from communication.messages import (
    AckMessage,
    ErrorMessage,
    PingMessage,
    PongMessage,
    SetStateMessage,
    SystemState,
    parse_message,
    serialize_message,
)
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FakeConveyorServer:
    """Simple WebSocket server for testing."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._server = None
        self._client: WebSocketServerProtocol | None = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def start(self) -> None:
        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )
        logger.info(f"🚀 Server started on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Server stopped")

    async def send_state(self, state: SystemState) -> bool:
        if self._client is None:
            logger.warning("❌ Cannot send: no client connected")
            return False

        try:
            msg = SetStateMessage(state=state)
            json_msg = serialize_message(msg)
            logger.info(f"📤 Sending: {json_msg}")
            await self._client.send(json_msg)
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.warning("❌ Connection closed while sending")
            self._client = None
            return False
        except AttributeError:
            logger.warning("❌ Client connection lost")
            self._client = None
            return False
        except Exception as e:
            logger.error(f"❌ Error sending: {e}")
            self._client = None
            return False

    async def send_ping(self) -> bool:
        if self._client is None:
            logger.warning("❌ Cannot send: no client connected")
            return False

        try:
            msg = PingMessage()
            json_msg = serialize_message(msg)
            logger.info(f"📤 Sending PING: {json_msg}")
            await self._client.send(json_msg)
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.warning("❌ Connection closed while sending")
            self._client = None
            return False
        except AttributeError:
            logger.warning("❌ Client connection lost")
            self._client = None
            return False
        except Exception as e:
            logger.error(f"❌ Error sending: {e}")
            self._client = None
            return False

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        if self._client is not None:
            logger.warning("Rejecting new client: already connected")
            await websocket.close(1008, "Only one client allowed")
            return

        self._client = websocket
        client_addr = websocket.remote_address
        logger.info(f"✅ Client connected from {client_addr}")

        try:
            async for raw_message in websocket:
                logger.info(f"📨 RAW MESSAGE RECEIVED: {raw_message}")

                try:
                    msg = parse_message(raw_message)
                    logger.info(f"📨 PARSED: {type(msg).__name__}")

                    if isinstance(msg, AckMessage):
                        logger.info(
                            f"   ✅ ACK for {msg.state.value}: success={msg.success}"
                        )
                    elif isinstance(msg, ErrorMessage):
                        logger.error(f"   ❌ ERROR: {msg.message}")
                    elif isinstance(msg, PongMessage):
                        logger.info("   🏓 PONG received")

                except ValueError as e:
                    logger.error(f"❌ Failed to parse: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔌 Client disconnected: {e}")
        finally:
            self._client = None
            logger.info("Client slot available")


async def keyboard_input(server: FakeConveyorServer, shutdown_event: asyncio.Event):
    """Handle keyboard input for sending commands."""
    logger.info("")
    logger.info("=" * 50)
    logger.info("  COMMANDS (type and press Enter):")
    logger.info("    r - Send RUNNING")
    logger.info("    f - Send FLIPPING")
    logger.info("    s - Send SORTING")
    logger.info("    x - Send STOPPED")
    logger.info("    p - Send PING")
    logger.info("    q - Quit")
    logger.info("=" * 50)
    logger.info("")

    loop = asyncio.get_running_loop()

    def blocking_input():
        try:
            return input("> ").strip().lower()
        except EOFError:
            return "q"

    while not shutdown_event.is_set():
        try:
            # Read from stdin - requires Enter to be pressed
            key = await loop.run_in_executor(None, blocking_input)

            if not key:
                continue
            elif key == "r":
                await server.send_state(SystemState.RUNNING)
            elif key == "f":
                await server.send_state(SystemState.FLIPPING)
            elif key == "s":
                await server.send_state(SystemState.SORTING)
            elif key == "x":
                await server.send_state(SystemState.STOPPED)
            elif key == "p":
                await server.send_ping()
            elif key == "q":
                shutdown_event.set()
                break
            else:
                logger.info(f"Unknown command: {key}")

        except Exception as e:
            logger.error(f"Input error: {e}")
            break


async def main():
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        pass

    server = FakeConveyorServer()

    try:
        await server.start()

        # Run keyboard handler
        await keyboard_input(server, shutdown_event)

    finally:
        await server.stop()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FAKE CONVEYOR SERVER - WebSocket Debug Tool")
    print("=" * 60 + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
