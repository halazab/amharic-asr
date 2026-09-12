"""Amharic ASR - Agent-ready speech recognition from scratch."""

__version__ = "0.1.0"

from .client import AmharicASR
from .models import TranscriptionResult, AudioConfig, VoiceControl

__all__ = ["AmharicASR", "TranscriptionResult", "AudioConfig", "VoiceControl"]
