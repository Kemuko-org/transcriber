import os
import tempfile
import validators
import whisper
import yt_dlp
import requests
import logging
import asyncio
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ping_service():
    url = os.getenv('DEPLOYED_URL')
    if url:
        try:
            response = requests.get(url + '/health', timeout=30)
            if response.status_code == 200:
                logging.info(f"Successfully pinged {url} - Status: {response.status_code}")
            else:
                logging.warning(f"Ping returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to ping {url}: {e}")


async def keep_alive_task():
    while True:
        ping_service()
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(keep_alive_task())
    yield
    task.cancel()


app = FastAPI(title="Media Transcriber",
              description="Transcribe audio/video from URLs using Whisper",
              lifespan=lifespan)


class TranscriptionRequest(BaseModel):
    url: str
    model_size: str = "base"


class TranscriptionResponse(BaseModel):
    text: str
    segments: list
    language: str


def validate_url(url: str) -> bool:
    return validators.url(url)


def download_media(url: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".%(ext)s", delete=False) as tmp_file:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tmp_file.name.replace('.%(ext)s', '.%(ext)s'),
            'extractaudio': True,
            'audioformat': 'mp3',
            'quiet': True,
            'no_warnings': True,
        }

        cookie_file_path = 'cookie.txt'
        youtube_cookies = os.environ.get('YOUTUBE_COOKIES')
        
        if os.path.exists(cookie_file_path):
            ydl_opts['cookiefile'] = cookie_file_path
        elif youtube_cookies:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as cookie_file:
                cookie_file.write("# Netscape HTTP Cookie File\n")
                cookie_file.write(youtube_cookies)
                cookie_file.flush()
                ydl_opts['cookiefile'] = cookie_file.name

        temp_cookie_file = None
        if 'cookiefile' in ydl_opts and ydl_opts['cookiefile'] != 'cookie.txt':
            temp_cookie_file = ydl_opts['cookiefile']

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return filename
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to download media: {str(e)}")
        finally:
            # Only delete temporary cookie files, not the local cookie.txt
            if temp_cookie_file and os.path.exists(temp_cookie_file):
                os.unlink(temp_cookie_file)


def transcribe_audio(file_path: str, model_size: str = "base") -> Dict[str, Any]:
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(file_path)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_media(request: TranscriptionRequest):
    if not validate_url(request.url):
        raise HTTPException(status_code=400, detail="Invalid URL provided")

    audio_file = None
    try:
        audio_file = download_media(request.url)

        result = transcribe_audio(audio_file, request.model_size)

        return TranscriptionResponse(
            text=result["text"],
            segments=result["segments"],
            language=result["language"]
        )

    finally:
        if audio_file and os.path.exists(audio_file):
            os.unlink(audio_file)


@app.get("/")
async def read_root():
    return {"message": "Media Transcriber API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
