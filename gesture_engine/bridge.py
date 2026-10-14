"""
WebSocket Real-time Bridge for Web & Browser Integration
Broadcasts zero-latency gesture events and cursor coordinates to any web application
(React, Next.js, Vue, or Vanilla HTML/JS) over local WebSockets.
"""

import asyncio
import json
import threading
import time
from typing import Set, Dict, Any, Optional


class GestureWebSocketBridge:
    """
    Thread-safe WebSocket server running in background daemon thread.
    Broadcasts real-time gesture actions and cursor positions to connected web clients.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False

    async def _client_handler(self, websocket: Any) -> None:
        """Handle incoming WebSocket connection from webpage or frontend client."""
        self.clients.add(websocket)
        try:
            welcome_msg = json.dumps({
                "type": "connection",
                "status": "connected",
                "message": "Connected to Gesture Control Engine",
                "version": "2.0.0"
            })
            await websocket.send(welcome_msg)
            await websocket.wait_closed()
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)

    def _run_event_loop(self) -> None:
        """Run asyncio event loop inside background daemon thread."""
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            print("[WARN] websockets library not installed. Web bridge disabled.")
            return

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def server_main():
            async with serve(self._client_handler, self.host, self.port):
                self._running = True
                print(f"[WEB BRIDGE] WebSocket server active at ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever

        try:
            self.loop.run_until_complete(server_main())
        except Exception as e:
            if self._running:
                print(f"[WEB BRIDGE] Server stopped: {e}")

    def start(self) -> "GestureWebSocketBridge":
        """Start the WebSocket server in a background thread."""
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True, name="GestureWebBridge")
        self.thread.start()
        time.sleep(0.08)
        return self

    def broadcast_event(self, action: str, cursor_xy: tuple, state: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Broadcast an interaction event (CLICK, SWIPE_*, HOLD_*, DRAG_*) to all connected web clients."""
        if not self.clients or not self.loop or not self.loop.is_running():
            return

        payload = {
            "type": "gesture",
            "action": action,
            "cursor": {"x": round(cursor_xy[0], 4), "y": round(cursor_xy[1], 4)},
            "state": state,
            "timestamp": round(time.time(), 3),
        }
        if extra:
            payload.update(extra)

        msg = json.dumps(payload)
        for ws in list(self.clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(msg), self.loop)
            except Exception:
                pass

    def broadcast_cursor(self, cursor_xy: tuple, state: str, pinch_ratio: float, palm_valid: bool) -> None:
        """Broadcast live cursor coordinates and posture to drive on-screen cursor on the webpage."""
        if not self.clients or not self.loop or not self.loop.is_running():
            return

        payload = {
            "type": "cursor",
            "cursor": {"x": round(cursor_xy[0], 4), "y": round(cursor_xy[1], 4)},
            "state": state,
            "pinch_ratio": round(pinch_ratio, 3),
            "palm_valid": palm_valid,
        }
        msg = json.dumps(payload)
        for ws in list(self.clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(msg), self.loop)
            except Exception:
                pass
