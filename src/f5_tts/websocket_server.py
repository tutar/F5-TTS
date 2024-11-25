from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from socket_server import TTSStreamingProcessor

import torch

import wave
import numpy as np
import io,os,tempfile



app = FastAPI()


# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="data/static"), name="static")

@app.get("/", include_in_schema=False)
async def get_index():
    return FileResponse(path="data/static/index.html")

@app.get("/audioStreamProcessor.js")
async def get_index():
    return FileResponse(path="data/static/audioStreamProcessor.js")

# Load the model and vocoder using the provided files
ckpt_file = "ckpts/model/model_1200000.safetensors"  # pointing your checkpoint "ckpts/model/model_1096.pt"
vocab_file = "ckpts/model/F5TTS_Base_vocab.txt"  # Add vocab file path if needed
ref_audio = "data/audio/leijun.MP3"  # add ref audio"./tests/ref_audio/reference.wav"
ref_text = "来参加广州车展小米发布会今天呢到场来了很多很多的朋友们也有不少我的老朋友们来捧场我跟大家介绍一位啊."

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

            # # 生成整块并返回
            # audio_chunk,target_sample_rate = processor.generate_stream1(text)
            # print(audio_chunk)

            # # 使用函数将final_wave数组转换为字节流
            # final_wave_bytes = array_to_wave(audio_chunk, target_sample_rate)
            # save_wave_to_cache(final_wave_bytes)

            final_wave_bytes = read_wav_file()

            # 然后通过websocket.send_bytes发送字节流
            await websocket.send_bytes(final_wave_bytes)            
            await websocket.send_bytes(b"END_OF_AUDIO")
            

            # # Generate and stream audio chunks
            # for audio_chunk in processor.generate_stream1(text):
            #     websocket.send_bytes(audio_chunk)

            # # Send end-of-audio signal
            # await websocket.send_bytes(b"END_OF_AUDIO")

    except WebSocketDisconnect:
        # 处理断开连接的情况
        print("Client disconnected")


def array_to_wave(final_wave, sample_rate):
    # 确保final_wave是16位整数格式
    audio_wave = (final_wave * 32767).astype(np.int16)
    
    # 将NumPy数组转换为字节流
    with io.BytesIO() as buffer:
        with wave.open(buffer, 'wb') as wave_file:
            wave_file.setnchannels(1)  # 单声道
            wave_file.setsampwidth(2)  # 每个样本2字节
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(audio_wave.tobytes())
        return buffer.getvalue()

def save_wave_to_cache(final_wave_bytes, cache_dir='data/audio'):
    # 创建缓存目录如果它不存在
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 创建一个临时文件名
    temp_file = tempfile.NamedTemporaryFile(mode='wb', dir=cache_dir, delete=False,suffix='.wav')
    
    # 将字节流写入临时文件
    temp_file.write(final_wave_bytes)
    temp_file.close()  # 关闭文件以确保数据被写入磁盘
    
    # 返回文件路径
    return temp_file.name

def read_wav_file(wav_file_path="data/audio/tmpa6mu5g4b.wav"):
    # 打开WAV文件
    with wave.open(wav_file_path, 'rb') as wav_file:
        # 获取WAV文件参数
        n_channels, sample_width, framerate, n_frames, comptype, compname = wav_file.getparams()
        
        # 读取所有帧
        frames = wav_file.readframes(wav_file.getnframes())
        return frames
