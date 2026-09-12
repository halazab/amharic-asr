"""Data models for Amharic ASR."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import uuid


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    WEBM = "webm"
    AAC = "aac"
    WMA = "wma"
    AIFF = "aiff"
    OPUS = "opus"


class OutputFormat(Enum):
    """Output formats for transcription."""
    TEXT = "text"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    TIMESTAMPED = "timestamped"


@dataclass
class VoiceControl:
    """Voice output control parameters for TTS-like processing."""
    
    # Audio preprocessing
    sample_rate: int = 16000
    channels: int = 1
    
    # Voice normalization
    normalize_volume: bool = True
    target_db: float = -20.0
    
    # Noise reduction
    noise_reduction: bool = True
    noise_reduction_level: float = 0.5  # 0.0 to 1.0
    
    # Voice activity detection
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.3  # seconds
    max_speech_duration: float = 30.0  # seconds
    
    # Audio trimming
    trim_silence: bool = True
    silence_threshold: float = -40  # dB
    silence_duration: float = 0.5  # seconds
    
    # Speed control (for playback)
    playback_speed: float = 1.0  # 0.5x to 2.0x
    
    # Pitch control (for analysis)
    pitch_shift: float = 0.0  # semitones (-12 to +12)
    
    # Augmentation (for training)
    speed_perturbation: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    noise_augmentation: bool = True
    rir_augmentation: bool = True


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    
    format: AudioFormat = AudioFormat.WAV
    voice_control: VoiceControl = field(default_factory=VoiceControl)
    
    # Advanced processing
    max_duration: float = 30.0
    min_duration: float = 0.5
    
    # Feature extraction
    feature_type: str = "raw"  # raw, fbank, mfcc
    n_features: int = 80
    
    # Streaming
    chunk_size_ms: int = 160
    overlap_ms: int = 40


@dataclass
class Timestamp:
    """Word-level timestamp."""
    
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    language: str = "am"
    confidence: float = 0.0
    
    # Timestamps
    timestamps: List[Timestamp] = field(default_factory=list)
    word_timestamps: bool = False
    
    # Audio info
    audio_duration: float = 0.0
    audio_format: str = "wav"
    sample_rate: int = 16000
    
    # Processing info
    processing_time_ms: float = 0.0
    model_version: str = "0.1.0"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "timestamps": [
                {
                    "word": t.word,
                    "start": t.start,
                    "end": t.end,
                    "confidence": t.confidence
                }
                for t in self.timestamps
            ],
            "audio_duration": self.audio_duration,
            "audio_format": self.audio_format,
            "sample_rate": self.sample_rate,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "metadata": self.metadata
        }
    
    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        srt_content = ""
        for i, ts in enumerate(self.timestamps, 1):
            start_h, start_m, start_s = self._seconds_to_srt_time(ts.start)
            end_h, end_m, end_s = self._seconds_to_srt_time(ts.end)
            srt_content += f"{i}\n"
            srt_content += f"{start_h:02d}:{start_m:02d}:{start_s:02.3f} --> "
            srt_content += f"{end_h:02d}:{end_m:02d}:{end_s:02.3f}\n"
            srt_content += f"{ts.word}\n\n"
        return srt_content
    
    def to_vtt(self) -> str:
        """Convert to WebVTT format."""
        vtt_content = "WEBVTT\n\n"
        for ts in self.timestamps:
            start = self._seconds_to_vtt_time(ts.start)
            end = self._seconds_to_vtt_time(ts.end)
            vtt_content += f"{start} --> {end}\n"
            vtt_content += f"{ts.word}\n\n"
        return vtt_content
    
    def _seconds_to_srt_time(self, seconds: float):
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return hours, minutes, secs
    
    def _seconds_to_vtt_time(self, seconds: float):
        """Convert seconds to VTT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class TranscriptionRequest:
    """Request for transcription."""
    
    audio: str  # Base64, URL, or file path
    language: str = "am"
    return_timestamps: bool = True
    return_confidence: bool = True
    output_format: OutputFormat = OutputFormat.JSON
    voice_control: Optional[VoiceControl] = None
    
    # Batch processing
    batch_id: Optional[str] = None
    batch_index: int = 0
    
    # Streaming
    stream: bool = False
    chunk_size_ms: int = 160
    
    # Webhook
    webhook_url: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJob:
    """Batch processing job."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"  # pending, processing, completed, failed
    
    # Input
    audio_files: List[str] = field(default_factory=list)
    total_files: int = 0
    
    # Progress
    processed_files: int = 0
    failed_files: int = 0
    
    # Results
    results: List[TranscriptionResult] = field(default_factory=list)
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Config
    language: str = "am"
    voice_control: Optional[VoiceControl] = None
    webhook_url: Optional[str] = None
    
    def progress(self) -> float:
        """Get progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "status": self.status,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "progress": self.progress(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }
