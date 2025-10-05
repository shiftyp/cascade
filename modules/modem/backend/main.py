"""CASCADE Modem Server - Main Entry Point

Python Server + React WebApp architecture for CASCADE HF digital modem.

Real-time multi-user communication with kernel-driven coordination.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import socketio

from config import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global state (will be moved to state/ modules later)
class CASCADEState:
    """Global server state"""
    def __init__(self):
        self.active_users = {}
        self.kernel_cache = {}
        self.conversations = {}
        self.current_net = None
        self.radio_connected = False
        self.audio_running = False


cascade_state = CASCADEState()


# SocketIO setup
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # Will restrict in production
    logger=logger,
    engineio_logger=logger if config.debug else False
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    logger.info("CASCADE Modem Server starting...")
    logger.info(f"Callsign: {config.callsign}")
    logger.info(f"Grid: {config.grid_square}")
    logger.info(f"Hardware: {config.hardware_tier}")
    logger.info(f"Max simultaneous users: {config.max_simultaneous_users}")

    # TODO: Initialize hardware interfaces
    # - Hamlib radio control
    # - Audio I/O
    # - PyTorch model loading

    # Start background tasks
    # asyncio.create_task(audio_receive_loop())
    # asyncio.create_task(decode_loop())

    logger.info(f"Server ready on http://{config.host}:{config.port}")

    yield

    # Cleanup on shutdown
    logger.info("CASCADE Modem Server shutting down...")
    # TODO: Close hardware connections
    # - Stop audio
    # - Close radio
    # - Save state


# FastAPI app
app = FastAPI(
    title="CASCADE Modem Server",
    description="Python backend for CASCADE HF digital modem",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for development (React dev server on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite, CRA
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SocketIO event handlers
@sio.event
async def connect(sid, environ):
    """Client connected"""
    logger.info(f"Client connected: {sid}")

    # Send initial state
    await sio.emit('connect_ack', {
        'sessionId': sid,
        'cascadeVersion': '0.1.0',
        'callsign': config.callsign,
        'gridSquare': config.grid_square
    }, room=sid)

    await sio.emit('network_state', {
        'activeUsers': len(cascade_state.active_users),
        'totalUsers': 1024,
        'myPatterns': [],  # TODO: Get from pattern assignment
        'kernelsCached': len(cascade_state.kernel_cache),
        'currentNet': None,
        'relayMode': False
    }, room=sid)


@sio.event
async def disconnect(sid):
    """Client disconnected"""
    logger.info(f"Client disconnected: {sid}")


@sio.event
async def send_message(sid, data):
    """Client wants to send a message"""
    logger.info(f"Send message request from {sid}: {data}")

    # TODO: Encode message via PyTorch model
    # TODO: Transmit via audio
    # For now, just echo back
    await sio.emit('message_sent', {
        'to': data.get('to'),
        'content': data.get('content'),
        'status': 'queued'
    }, room=sid)


@sio.event
async def set_frequency(sid, data):
    """Client wants to change frequency"""
    freq = data.get('frequency')
    logger.info(f"Frequency change request: {freq} Hz")

    # TODO: Use Hamlib to set frequency
    config.frequency = freq

    await sio.emit('frequency_changed', {
        'frequency': freq,
        'status': 'ok'
    }, room=sid)


# Mount SocketIO to FastAPI
socket_app = socketio.ASGIApp(sio, app)


# REST API Routes (will be moved to api/routes.py)
@app.get("/")
async def root():
    """Health check / API info"""
    return {
        "service": "CASCADE Modem Server",
        "version": "0.1.0",
        "callsign": config.callsign,
        "status": "running"
    }


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "radio_connected": cascade_state.radio_connected,
        "audio_running": cascade_state.audio_running,
        "active_users": len(cascade_state.active_users)
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "callsign": config.callsign,
        "gridSquare": config.grid_square,
        "frequency": config.frequency,
        "mode": config.mode,
        "hardwareTier": config.hardware_tier,
        "maxSimultaneousUsers": config.max_simultaneous_users
    }


@app.post("/api/config")
async def update_config(data: dict):
    """Update configuration"""
    # TODO: Validate and update config
    logger.info(f"Config update request: {data}")
    return {"status": "updated"}


@app.get("/api/network/topology")
async def get_network_topology():
    """Get current network topology"""
    return {
        "nodes": [],  # TODO: Build from active_users
        "links": [],  # TODO: Build from known connections
        "activeUsers": len(cascade_state.active_users),
        "totalCapacity": 1024
    }


# For production: serve React build
# app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        socket_app,
        host=config.host,
        port=config.port,
        log_level="debug" if config.debug else "info"
    )
