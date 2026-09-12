"""Audio processing utilities for multi-format support and voice control."""

import io
import struct
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .models import AudioConfig, VoiceControl, AudioFormat


@dataclass
class AudioProcessor:
    """Process audio in multiple formats with voice control."""
    
    config: AudioConfig = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = AudioConfig()
    
    def detect_format(self, data: bytes) -> AudioFormat:
        """Detect audio format from magic bytes."""
        if data[:4] == b'RIFF':
            return AudioFormat.WAV
        elif data[:3] == b'ID3' or data[:2] == b'\xff\xfb':
            return AudioFormat.MP3
        elif data[:4] == b'fLaC':
            return AudioFormat.FLAC
        elif data[:4] == b'OggS':
            return AudioFormat.OGG
        elif len(data) > 4 and data[4:8] == b'ftyp':
            return AudioFormat.M4A
        elif data[:4] == b'\x1a\x45\xdf\xa3':
            return AudioFormat.WEBM
        elif data[:4] == b'FORM':
            return AudioFormat.AIFF
        elif data[:4] == b'OGGS' or data[:8] == b'OpusHead':
            return AudioFormat.OPUS
        else:
            return AudioFormat.WAV  # Default
    
    def read_audio(self, data: bytes, format: Optional[AudioFormat] = None) -> Tuple[np.ndarray, int]:
        """Read audio data and return as numpy array."""
        if format is None:
            format = self.detect_format(data)
        
        # Use soundfile for reading
        import soundfile as sf
        
        # Convert bytes to file-like object
        audio_file = io.BytesIO(data)
        
        # Read based on format
        try:
            audio, sample_rate = sf.read(audio_file, format=format.value.upper())
        except Exception:
            # Fallback: try with wavpy
            import wave
            audio_file.seek(0)
            with wave.open(audio_file, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
                
                # Convert to numpy array
                if sampwidth == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                elif sampwidth == 4:
                    audio = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    audio = np.frombuffer(raw_data, dtype=np.float32)
                
                # Reshape for channels
                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        return audio, sample_rate
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        # Simple resampling using linear interpolation
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        # Interpolate
        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return resampled
    
    def normalize_volume(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio volume to target dB level."""
        # Calculate current dB level
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        
        current_db = 20 * np.log10(rms)
        
        # Calculate gain
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain
        
        # Clip to prevent distortion
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def reduce_noise(self, audio: np.ndarray, level: float = 0.5) -> np.ndarray:
        """Simple noise reduction using spectral gating."""
        if level == 0:
            return audio
        
        # FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Simple noise gate based on magnitude
        threshold = np.mean(magnitude) * (1 - level)
        mask = magnitude > threshold
        
        # Apply mask
        magnitude = magnitude * mask
        
        # Reconstruct
        fft_filtered = magnitude * np.exp(1j * phase)
        audio_filtered = np.fft.irfft(fft_filtered)
        
        return audio_filtered
    
    def trim_silence(self, audio: np.ndarray, sample_rate: int, 
                     threshold_db: float = -40, min_silence_duration: float = 0.5) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        # Convert threshold to amplitude
        threshold = 10 ** (threshold_db / 20)
        
        # Find non-silent regions
        is_speech = np.abs(audio) > threshold
        
        # Find first and last speech indices
        speech_indices = np.where(is_speech)[0]
        
        if len(speech_indices) == 0:
            return audio
        
        # Add small padding
        padding = int(sample_rate * 0.1)  # 100ms padding
        start_idx = max(0, speech_indices[0] - padding)
        end_idx = min(len(audio), speech_indices[-1] + padding)
        
        return audio[start_idx:end_idx]
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int,
                              threshold: float = 0.5, min_duration: float = 0.3) -> List[Tuple[float, float]]:
        """Detect voice activity segments."""
        # Calculate energy in short windows
        window_size = int(sample_rate * 0.02)  # 20ms windows
        hop_size = window_size // 2
        
        energies = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            energy = np.sqrt(np.mean(window ** 2))
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Normalize energies
        if np.max(energies) > 0:
            energies = energies / np.max(energies)
        
        # Find segments above threshold
        is_speech = energies > threshold
        
        # Find contiguous speech regions
        segments = []
        in_segment = False
        start_time = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_size / sample_rate
            if speech and not in_segment:
                start_time = time
                in_segment = True
            elif not speech and in_segment:
                duration = time - start_time
                if duration >= min_duration:
                    segments.append((start_time, time))
                in_segment = False
        
        # Handle last segment
        if in_segment:
            duration = len(audio) / sample_rate - start_time
            if duration >= min_duration:
                segments.append((start_time, len(audio) / sample_rate))
        
        return segments
    
    def apply_voice_control(self, audio: np.ndarray, sample_rate: int,
                           voice_control: VoiceControl) -> Tuple[np.ndarray, int]:
        """Apply all voice control parameters."""
        # Resample if needed
        if sample_rate != voice_control.sample_rate:
            audio = self.resample(audio, sample_rate, voice_control.sample_rate)
            sample_rate = voice_control.sample_rate
        
        # Normalize volume
        if voice_control.normalize_volume:
            audio = self.normalize_volume(audio, voice_control.target_db)
        
        # Noise reduction
        if voice_control.noise_reduction:
            audio = self.reduce_noise(audio, voice_control.noise_reduction_level)
        
        # Trim silence
        if voice_control.trim_silence:
            audio = self.trim_silence(audio, sample_rate, 
                                      voice_control.silence_threshold,
                                      voice_control.silence_duration)
        
        return audio, sample_rate
    
    def prepare_for_model(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare audio for model input."""
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = self.resample(audio, sample_rate, 16000)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        return audio
    
    def split_audio(self, audio: np.ndarray, sample_rate: int,
                   chunk_duration: float = 30.0, overlap: float = 0.0) -> List[np.ndarray]:
        """Split audio into chunks."""
        chunk_size = int(chunk_duration * sample_rate)
        overlap_size = int(overlap * sample_rate)
        hop_size = chunk_size - overlap_size
        
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]
            
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            chunks.append(chunk)
            start += hop_size
        
        return chunks
    
    def get_audio_info(self, data: bytes) -> dict:
        """Get audio file information."""
        format = self.detect_format(data)
        audio, sample_rate = self.read_audio(data, format)
        
        return {
            "format": format.value,
            "sample_rate": sample_rate,
            "channels": 1 if len(audio.shape) == 1 else audio.shape[1],
            "duration": len(audio) / sample_rate,
            "samples": len(audio),
            "bit_depth": 32,
            "file_size": len(data)
        }
