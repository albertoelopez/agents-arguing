import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.web.websocket_handler import DebateWebSocket
from src.realtime.streaming import StreamingDebateSession
from src.config import settings


app = FastAPI(
    title="Agents Arguing",
    description="AI agents debating with real-time voice and video",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: dict[str, StreamingDebateSession] = {}
ws_handler = DebateWebSocket()


class DebateConfig(BaseModel):
    topic: str
    pro_name: str = "Alex"
    pro_personality: str = "Optimistic, data-driven, focuses on innovation"
    con_name: str = "Jordan"
    con_personality: str = "Skeptical, philosophical, emphasizes risks"
    num_rounds: int = 3
    enable_audio: bool = False


class DebateStatus(BaseModel):
    session_id: str
    is_running: bool
    topic: str
    current_turn: int


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agents Arguing</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
            h1 { color: #4CAF50; }
            a { color: #64B5F6; }
            .card { background: #16213e; padding: 20px; border-radius: 8px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🎭 Agents Arguing</h1>
        <div class="card">
            <h2>API Endpoints</h2>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/ui">Gradio Web UI</a></li>
                <li>WebSocket: <code>ws://localhost:8000/ws/debate/{session_id}</code></li>
            </ul>
        </div>
        <div class="card">
            <h2>Quick Start</h2>
            <p>1. POST to <code>/api/debate/start</code> with your topic</p>
            <p>2. Connect to WebSocket to receive streaming updates</p>
            <p>3. Or use the Gradio UI at <a href="/ui">/ui</a></p>
        </div>
    </body>
    </html>
    """


@app.get("/live", response_class=HTMLResponse)
async def live_client():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse(content="<h1>Static files not found</h1>", status_code=404)


@app.post("/api/debate/start", response_model=DebateStatus)
async def start_debate(config: DebateConfig):
    import uuid
    session_id = str(uuid.uuid4())[:8]

    session = StreamingDebateSession(
        topic=config.topic,
        pro_name=config.pro_name,
        pro_personality=config.pro_personality,
        con_name=config.con_name,
        con_personality=config.con_personality,
        num_rounds=config.num_rounds,
        enable_audio=config.enable_audio,
    )

    await session.initialize()
    active_sessions[session_id] = session

    return DebateStatus(
        session_id=session_id,
        is_running=False,
        topic=config.topic,
        current_turn=0,
    )


@app.get("/api/debate/{session_id}/status", response_model=DebateStatus)
async def get_status(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    return DebateStatus(
        session_id=session_id,
        is_running=session.is_running,
        topic=session.topic,
        current_turn=len(session._debate_manager.result.turns) if session._debate_manager else 0,
    )


@app.post("/api/debate/{session_id}/stop")
async def stop_debate(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    session.stop()
    return {"status": "stopping", "session_id": session_id}


@app.delete("/api/debate/{session_id}")
async def delete_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions.pop(session_id)
    await session.shutdown()
    return {"status": "deleted", "session_id": session_id}


@app.websocket("/ws/debate/{session_id}")
async def websocket_debate(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.send_json({"type": "error", "data": {"error": "Session not found"}})
        await websocket.close()
        return

    session = active_sessions[session_id]

    try:
        async for event in session.stream_debate():
            await websocket.send_text(event.to_json())
    except WebSocketDisconnect:
        session.stop()
    except Exception as e:
        await websocket.send_json({"type": "error", "data": {"error": str(e)}})
    finally:
        await websocket.close()


@app.websocket("/ws/debate/new")
async def websocket_new_debate(websocket: WebSocket):
    await websocket.accept()

    try:
        config_data = await websocket.receive_json()

        session = StreamingDebateSession(
            topic=config_data.get("topic", "AI benefits humanity"),
            pro_name=config_data.get("pro_name", "Alex"),
            pro_personality=config_data.get("pro_personality", "Optimistic"),
            con_name=config_data.get("con_name", "Jordan"),
            con_personality=config_data.get("con_personality", "Skeptical"),
            num_rounds=config_data.get("num_rounds", 3),
            enable_audio=config_data.get("enable_audio", False),
        )

        await session.initialize()

        async for event in session.stream_debate():
            await websocket.send_text(event.to_json())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "data": {"error": str(e)}})
    finally:
        await websocket.close()


def main():
    uvicorn.run(
        "src.web.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
