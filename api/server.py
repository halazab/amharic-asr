"""FastAPI server for Amharic ASR."""

import os
import uuid
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from amharic_asr import AmharicASR, AudioConfig, VoiceControl
from amharic_asr.models import AudioFormat, OutputFormat


# Initialize app
app = FastAPI(
    title="Amharic ASR API",
    description="State-of-the-art Amharic speech recognition with agent tool calling",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ASR client
asr_client = AmharicASR()

# In-memory job storage (use Redis in production)
batch_jobs: Dict[str, Dict[str, Any]] = {}


# Request/Response models
class TranscriptionResponse(BaseModel):
    id: str
    text: str
    language: str
    confidence: float
    timestamps: List[Dict[str, Any]]
    audio_duration: float
    audio_format: str
    processing_time_ms: float


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    total_files: int
    processed_files: int
    progress: float


class VoiceControlRequest(BaseModel):
    normalize_volume: bool = True
    target_db: float = -20.0
    noise_reduction: bool = True
    noise_reduction_level: float = 0.5
    trim_silence: bool = True
    silence_threshold: float = -40
    silence_duration: float = 0.5
    sample_rate: int = 16000


class ToolDefinitionResponse(BaseModel):
    tools: List[Dict[str, Any]]


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "amharic-asr",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat()
    }


# Transcription endpoint
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="am"),
    return_timestamps: bool = Form(default=True),
    return_confidence: bool = Form(default=True),
    output_format: str = Form(default="json"),
    normalize_volume: bool = Form(default=True),
    noise_reduction: bool = Form(default=True),
    trim_silence: bool = Form(default=True),
    sample_rate: int = Form(default=16000)
):
    """
    Transcribe an audio file.
    
    Supports: WAV, MP3, FLAC, OGG, M4A, WebM, AAC, WMA, AIFF, OPUS
    """
    # Read audio data
    audio_data = await file.read()
    
    # Create voice control
    voice_control = VoiceControl(
        normalize_volume=normalize_volume,
        noise_reduction=noise_reduction,
        trim_silence=trim_silence,
        sample_rate=sample_rate
    )
    
    # Transcribe
    result = asr_client.transcribe(
        audio_data,
        language=language,
        return_timestamps=return_timestamps,
        return_confidence=return_confidence,
        voice_control=voice_control
    )
    
    return TranscriptionResponse(
        id=result.id,
        text=result.text,
        language=result.language,
        confidence=result.confidence,
        timestamps=[t.__dict__ for t in result.timestamps],
        audio_duration=result.audio_duration,
        audio_format=result.audio_format,
        processing_time_ms=result.processing_time_ms
    )


# Transcription from URL
class URLTranscriptionRequest(BaseModel):
    url: str
    language: str = "am"
    return_timestamps: bool = True
    return_confidence: bool = True
    voice_control: Optional[VoiceControlRequest] = None


@app.post("/transcribe_url", response_model=TranscriptionResponse)
async def transcribe_from_url(request: URLTranscriptionRequest):
    """Transcribe audio from URL."""
    # Create voice control
    voice_control = None
    if request.voice_control:
        voice_control = VoiceControl(**request.voice_control.dict())
    
    # Transcribe
    result = asr_client.transcribe(
        request.url,
        language=request.language,
        return_timestamps=request.return_timestamps,
        return_confidence=request.return_confidence,
        voice_control=voice_control
    )
    
    return TranscriptionResponse(
        id=result.id,
        text=result.text,
        language=result.language,
        confidence=result.confidence,
        timestamps=[t.__dict__ for t in result.timestamps],
        audio_duration=result.audio_duration,
        audio_format=result.audio_format,
        processing_time_ms=result.processing_time_ms
    )


# Batch transcription
class BatchRequest(BaseModel):
    audio_files: List[str]
    language: str = "am"
    max_concurrent: int = 5
    voice_control: Optional[VoiceControlRequest] = None


@app.post("/transcribe_batch", response_model=BatchJobResponse)
async def transcribe_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """Start batch transcription job."""
    job_id = str(uuid.uuid4())
    
    # Store job
    batch_jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "total_files": len(request.audio_files),
        "processed_files": 0,
        "results": [],
        "created_at": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(
        process_batch,
        job_id=job_id,
        audio_files=request.audio_files,
        language=request.language,
        voice_control=request.voice_control
    )
    
    return BatchJobResponse(
        job_id=job_id,
        status="pending",
        total_files=len(request.audio_files),
        processed_files=0,
        progress=0.0
    )


async def process_batch(
    job_id: str,
    audio_files: List[str],
    language: str,
    voice_control: Optional[VoiceControlRequest]
):
    """Process batch job in background."""
    job = batch_jobs[job_id]
    job["status"] = "processing"
    job["started_at"] = datetime.now().isoformat()
    
    try:
        # Create voice control
        vc = None
        if voice_control:
            vc = VoiceControl(**voice_control.dict())
        
        # Process each file
        for audio_file in audio_files:
            try:
                result = asr_client.transcribe(
                    audio_file,
                    language=language,
                    voice_control=vc
                )
                job["results"].append(result.to_dict())
            except Exception as e:
                job["results"].append({
                    "error": str(e),
                    "file": audio_file
                })
            
            job["processed_files"] += 1
        
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


# Get batch job status
@app.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Get batch job status."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    return BatchJobResponse(
        job_id=job["id"],
        status=job["status"],
        total_files=job["total_files"],
        processed_files=job["processed_files"],
        progress=(job["processed_files"] / job["total_files"] * 100) if job["total_files"] > 0 else 0
    )


# Get batch job results
@app.get("/batch/{job_id}/results")
async def get_batch_results(job_id: str):
    """Get batch job results."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=202, detail="Job not completed yet")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "results": job["results"]
    }


# Tool definitions for agents
@app.get("/tools", response_model=ToolDefinitionResponse)
async def get_tool_definitions():
    """Get tool definitions for AI agent frameworks."""
    return ToolDefinitionResponse(
        tools=[
            asr_client.get_tool_definition(),
            asr_client.get_batch_tool_definition()
        ]
    )


# OpenAPI tool definition
@app.get("/tools/openai")
async def get_openai_tools():
    """Get OpenAI-compatible tool definitions."""
    return {
        "tools": [
            asr_client.get_tool_definition(),
            asr_client.get_batch_tool_definition()
        ]
    }


# Audio format support info
@app.get("/formats")
async def get_supported_formats():
    """Get list of supported audio formats."""
    return {
        "input_formats": [
            {"format": "wav", "extension": ".wav", "mime": "audio/wav"},
            {"format": "mp3", "extension": ".mp3", "mime": "audio/mpeg"},
            {"format": "flac", "extension": ".flac", "mime": "audio/flac"},
            {"format": "ogg", "extension": ".ogg", "mime": "audio/ogg"},
            {"format": "m4a", "extension": ".m4a", "mime": "audio/mp4"},
            {"format": "webm", "extension": ".webm", "mime": "audio/webm"},
            {"format": "aac", "extension": ".aac", "mime": "audio/aac"},
            {"format": "wma", "extension": ".wma", "mime": "audio/x-ms-wma"},
            {"format": "aiff", "extension": ".aiff", "mime": "audio/aiff"},
            {"format": "opus", "extension": ".opus", "mime": "audio/opus"}
        ],
        "output_formats": ["json", "srt", "vtt"],
        "sample_rates": [8000, 16000, 22050, 44100, 48000]
    }


# Voice control options
@app.get("/voice-control")
async def get_voice_control_options():
    """Get available voice control options."""
    return {
        "normalization": {
            "normalize_volume": {"type": "bool", "default": True},
            "target_db": {"type": "float", "default": -20.0, "min": -50, "max": 0}
        },
        "noise_reduction": {
            "enabled": {"type": "bool", "default": True},
            "level": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
        },
        "voice_activity_detection": {
            "enabled": {"type": "bool", "default": True},
            "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "min_speech_duration": {"type": "float", "default": 0.3}
        },
        "trimming": {
            "trim_silence": {"type": "bool", "default": True},
            "silence_threshold": {"type": "float", "default": -40},
            "silence_duration": {"type": "float", "default": 0.5}
        },
        "resampling": {
            "sample_rate": {"type": "int", "default": 16000, "options": [8000, 16000, 22050, 44100, 48000]}
        }
    }


# Metrics
@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return {
        "total_transcriptions": sum(1 for j in batch_jobs.values() if j["status"] == "completed"),
        "total_batch_jobs": len(batch_jobs),
        "active_jobs": sum(1 for j in batch_jobs.values() if j["status"] == "processing"),
        "average_processing_time_ms": 0.0,  # Would track in production
        "model_version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
