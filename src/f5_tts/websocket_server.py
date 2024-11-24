from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from socket_server import TTSStreamingProcessor

import torch



app = FastAPI()

# Load the model and vocoder using the provided files
ckpt_file = ""  # pointing your checkpoint "ckpts/model/model_1096.pt"
vocab_file = ""  # Add vocab file path if needed
ref_audio = ""  # add ref audio"./tests/ref_audio/reference.wav"
ref_text = ""

# Initialize the processor with the model and vocoder
processor = TTSStreamingProcessor(
    ckpt_file=ckpt_file,
    vocab_file=vocab_file,
    ref_audio=ref_audio,
    ref_text=ref_text,
    dtype=torch.float32,
)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:

        print("Client connected")

        while True:
            # 接收客户端消息
            text = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {text}")

            # Generate and stream audio chunks
            for audio_chunk in processor.generate_stream(text):
                await websocket.send_bytes(audio_chunk)

            # Send end-of-audio signal
            await websocket.send_bytes(b"END_OF_AUDIO")

    except WebSocketDisconnect:
        # 处理断开连接的情况
        print("Client disconnected")




# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def get_index():
    return FileResponse(path="static/index.html")

