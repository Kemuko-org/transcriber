import os
import tempfile
import validators
import whisper
import yt_dlp
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Media Transcriber",
              description="Transcribe audio/video from URLs using Whisper")


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

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return filename
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to download media: {str(e)}")


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
