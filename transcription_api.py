from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import whisper
import requests
import tempfile
import os
from pathlib import Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper Transcription API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model (using base model for balance of speed/accuracy)
logger.info("Loading Whisper model...")
model = whisper.load_model("base")
logger.info("Whisper model loaded successfully")


class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    language: str = "auto"  # Optional language hint


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list = []


@app.get("/")
async def root():
    return {"message": "Whisper Transcription API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "whisper-base"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcribe audio from a URL using Whisper
    """
    try:
        logger.info(f"Starting transcription for URL: {request.audio_url}")

        # Download audio file from URL
        response = requests.get(str(request.audio_url), timeout=30)
        response.raise_for_status()

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        try:
            # Transcribe using Whisper
            logger.info("Starting Whisper transcription...")

            # Set language if specified (None for auto-detection)
            language = None if request.language == "auto" else request.language

            result = model.transcribe(
                temp_file_path,
                language=language,
                verbose=True
            )

            logger.info("Transcription completed successfully")

            # Extract segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })

            return TranscriptionResponse(
                text=result["text"].strip(),
                language=result["language"],
                duration=result.get("duration", 0),
                segments=segments
            )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except requests.RequestException as e:
        logger.error(f"Error downloading audio: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Failed to download audio: {str(e)}")

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe-simple")
async def transcribe_simple(request: TranscriptionRequest):
    """
    Simple transcription endpoint that returns only the text
    """
    try:
        result = await transcribe_audio(request)
        return {"text": result.text, "language": result.language}
    except Exception as e:
        raise e

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "transcription_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
