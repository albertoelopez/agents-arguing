import asyncio
import json
from typing import Callable
from dataclasses import dataclass

from fastapi import WebSocket


@dataclass
class WebSocketClient:
    websocket: WebSocket
    session_id: str
    is_active: bool = True


class DebateWebSocket:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocketClient]] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> WebSocketClient:
        await websocket.accept()
        client = WebSocketClient(websocket=websocket, session_id=session_id)

        if session_id not in self.active_connections:
            self.active_connections[session_id] = []

        self.active_connections[session_id].append(client)
        return client

    def disconnect(self, client: WebSocketClient) -> None:
        client.is_active = False
        if client.session_id in self.active_connections:
            self.active_connections[client.session_id] = [
                c for c in self.active_connections[client.session_id]
                if c != client
            ]

    async def broadcast_to_session(self, session_id: str, message: dict) -> None:
        if session_id not in self.active_connections:
            return

        disconnected = []
        for client in self.active_connections[session_id]:
            try:
                await client.websocket.send_json(message)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            self.disconnect(client)

    async def send_to_client(self, client: WebSocketClient, message: dict) -> bool:
        try:
            await client.websocket.send_json(message)
            return True
        except Exception:
            self.disconnect(client)
            return False

    def get_connection_count(self, session_id: str) -> int:
        return len(self.active_connections.get(session_id, []))

    def get_all_sessions(self) -> list[str]:
        return list(self.active_connections.keys())
