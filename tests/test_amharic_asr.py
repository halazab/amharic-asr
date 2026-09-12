"""Tests for Amharic ASR SDK."""

import pytest
import numpy as np
from pathlib import Path

from amharic_asr import AmharicASR, AudioConfig, VoiceControl
from amharic_asr.models import AudioFormat, OutputFormat, TranscriptionResult
from amharic_asr.audio import AudioProcessor


class TestAudioProcessor:
    """Test audio processing utilities."""
    
    def test_detect_format_wav(self):
        """Test WAV format detection."""
        processor = AudioProcessor()
        
        # WAV magic bytes
        wav_header = b'RIFF' + b'\x00' * 4 + b'WAVE'
        format = processor.detect_format(wav_header)
        assert format == AudioFormat.WAV
    
    def test_detect_format_mp3(self):
        """Test MP3 format detection."""
        processor = AudioProcessor()
        
        # MP3 ID3 header
        mp3_header = b'ID3' + b'\x00' * 4
        format = processor.detect_format(mp3_header)
        assert format == AudioFormat.MP3
    
    def test_detect_format_flac(self):
        """Test FLAC format detection."""
        processor = AudioProcessor()
        
        # FLAC magic bytes
        flac_header = b'fLaC'
        format = processor.detect_format(flac_header)
        assert format == AudioFormat.FLAC
    
    def test_normalize_volume(self):
        """Test volume normalization."""
        processor = AudioProcessor()
        
        # Create silent audio
        audio = np.zeros(16000)
        
        # Should not crash on silent audio
        normalized = processor.normalize_volume(audio, target_db=-20.0)
        assert len(normalized) == len(audio)
    
    def test_trim_silence(self):
        """Test silence trimming."""
        processor = AudioProcessor()
        
        # Create audio with silence at start and end
        silence = np.zeros(8000)  # 0.5s silence
        speech = np.random.randn(8000) * 0.1  # 0.5s speech
        audio = np.concatenate([silence, speech, silence])
        
        trimmed = processor.trim_silence(audio, 16000)
        
        # Should be shorter
        assert len(trimmed) < len(audio)


class TestVoiceControl:
    """Test voice control parameters."""
    
    def test_default_values(self):
        """Test default voice control values."""
        vc = VoiceControl()
        
        assert vc.sample_rate == 16000
        assert vc.channels == 1
        assert vc.normalize_volume is True
        assert vc.noise_reduction is True
        assert vc.trim_silence is True
    
    def test_custom_values(self):
        """Test custom voice control values."""
        vc = VoiceControl(
            sample_rate=22050,
            normalize_volume=False,
            noise_reduction_level=0.8
        )
        
        assert vc.sample_rate == 22050
        assert vc.normalize_volume is False
        assert vc.noise_reduction_level == 0.8


class TestTranscriptionResult:
    """Test transcription result."""
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = TranscriptionResult(
            text="ሰላም ዓለም",
            confidence=0.95,
            language="am"
        )
        
        d = result.to_dict()
        assert d["text"] == "ሰላም ዓለም"
        assert d["confidence"] == 0.95
        assert d["language"] == "am"
    
    def test_to_srt(self):
        """Test SRT format conversion."""
        from amharic_asr.models import Timestamp
        
        result = TranscriptionResult(
            text="ሰላም",
            timestamps=[
                Timestamp(word="ሰላም", start=0.0, end=1.0, confidence=0.95)
            ]
        )
        
        srt = result.to_srt()
        assert "ሰላም" in srt
        assert "00:00:00" in srt
    
    def test_to_vtt(self):
        """Test VTT format conversion."""
        from amharic_asr.models import Timestamp
        
        result = TranscriptionResult(
            text="ሰላም",
            timestamps=[
                Timestamp(word="ሰላም", start=0.0, end=1.0, confidence=0.95)
            ]
        )
        
        vtt = result.to_vtt()
        assert "WEBVTT" in vtt
        assert "ሰላም" in vtt


class TestAmharicASR:
    """Test main ASR client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = AmharicASR()
        assert client.api_url == "http://localhost:8000"
    
    def test_tool_definition(self):
        """Test tool definition generation."""
        client = AmharicASR()
        tool_def = client.get_tool_definition()
        
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "transcribe_amharic"
        assert "audio" in tool_def["function"]["parameters"]["properties"]
    
    def test_batch_tool_definition(self):
        """Test batch tool definition generation."""
        client = AmharicASR()
        tool_def = client.get_batch_tool_definition()
        
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "transcribe_amharic_batch"


class TestAudioConfig:
    """Test audio configuration."""
    
    def test_default_config(self):
        """Test default audio configuration."""
        config = AudioConfig()
        
        assert config.format == AudioFormat.WAV
        assert config.max_duration == 30.0
        assert config.voice_control is not None
    
    def test_custom_config(self):
        """Test custom audio configuration."""
        vc = VoiceControl(sample_rate=22050)
        config = AudioConfig(
            format=AudioFormat.MP3,
            max_duration=60.0,
            voice_control=vc
        )
        
        assert config.format == AudioFormat.MP3
        assert config.max_duration == 60.0
        assert config.voice_control.sample_rate == 22050


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
