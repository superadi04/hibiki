import modal
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from datetime import datetime
import websockets
import asyncio

app = modal.App("hibiki-rs-realtime")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "apt-get update && apt-get install -y "
        "curl git build-essential pkg-config libssl-dev libclang-dev cmake "
        "ffmpeg libsox-dev libsox-fmt-all",
        "curl https://sh.rustup.rs -sSf | bash -s -- -y",
        ". $HOME/.cargo/env && cargo --version",
        "git clone https://github.com/superadi04/hibiki.git /hibiki",
        "cd /hibiki/hibiki-rs && . $HOME/.cargo/env && CUDA_COMPUTE_CAP=86 cargo build --release --bin server",
        force_build=True
    )
    .pip_install("fastapi", "uvicorn", "websockets")
)

web_app = FastAPI()

@web_app.get("/")
async def root():
    return {"message": "Welcome to Real-Time Hibiki API. Connect to /ws for audio streaming."}

@web_app.websocket("/ws")
async def websocket_proxy(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(f"[{datetime.now().strftime('%H:%M:%S')}] Connected. Streaming to Hibiki...")

    uri = "ws://localhost:8080/ws"  # The Rust server WebSocket endpoint inside Modal

    try:
        async with websockets.connect(uri) as rust_socket:
            while True:
                msg = await websocket.receive()

                if "bytes" in msg:
                    audio_chunk = msg["bytes"]
                    await rust_socket.send(audio_chunk)
                else:
                    await websocket.send_text("Expected audio chunk.")
                    continue

                # Forward response from Rust back to client
                response = await rust_socket.recv()
                if isinstance(response, bytes):
                    await websocket.send_bytes(response)
                elif isinstance(response, str):
                    await websocket.send_text(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(f"Error: {e}")

@app.function(image=image, gpu="A10G", timeout=600)
def run_hibiki_server():
    import subprocess
    subprocess.run(["/hibiki/hibiki-rs/target/release/server"])

@app.function(image=image, gpu="A10G", timeout=600)
@modal.asgi_app()
def serve_app():
    return web_app
