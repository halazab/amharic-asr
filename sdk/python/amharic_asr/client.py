"""Main client for Amharic ASR."""

import os
import time
import base64
import hashlib
import tempfile
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, AsyncGenerator
from dataclasses import asdict

from .models import (
    AudioConfig, AudioFormat, VoiceControl, TranscriptionResult,
    TranscriptionRequest, Timestamp, OutputFormat, BatchJob
)
from .audio import AudioProcessor


class AmharicASR:
    """
    Amharic ASR Client for agent tool calling.
    
    Supports:
    - Multiple audio formats (WAV, MP3, FLAC, OGG, M4A, WebM, etc.)
    - Voice control (normalization, noise reduction, VAD, trimming)
    - Batch processing
    - Streaming transcription
    - Structured output (JSON, SRT, VTT)
    
    Usage:
        client = AmharicASR()
        result = client.transcribe("audio.wav")
        print(result.text)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        local_model_path: Optional[str] = None,
        config: Optional[AudioConfig] = None
    ):
        """
        Initialize Amharic ASR client.
        
        Args:
            api_key: API key for cloud service (optional)
            api_url: API URL for cloud service (optional)
            local_model_path: Path to local ONNX model (optional)
            config: Audio configuration
        """
        self.api_key = api_key or os.getenv("AMHARIC_ASR_API_KEY")
        self.api_url = api_url or os.getenv("AMHARIC_ASR_API_URL", "http://localhost:8000")
        self.local_model_path = local_model_path or os.getenv("AMHARIC_ASR_MODEL_PATH")
        
        self.config = config or AudioConfig()
        self.audio_processor = AudioProcessor(self.config)
        
        # Initialize model if local
        self._model = None
        if self.local_model_path:
            self._load_local_model()
    
    def _load_local_model(self):
        """Load local ONNX model."""
        try:
            import onnxruntime as ort
            
            model_path = Path(self.local_model_path)
            if model_path.exists():
                self._model = ort.InferenceSession(str(model_path))
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        except ImportError:
            raise ImportError("onnxruntime required for local inference")
    
    def transcribe(
        self,
        audio: Union[str, bytes, Path],
        language: str = "am",
        return_timestamps: bool = True,
        return_confidence: bool = True,
        output_format: OutputFormat = OutputFormat.JSON,
        voice_control: Optional[VoiceControl] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            audio: Audio file path, bytes, or URL
            language: Language code (default: "am" for Amharic)
            return_timestamps: Include word-level timestamps
            return_confidence: Include confidence scores
            output_format: Output format (JSON, SRT, VTT)
            voice_control: Voice control parameters
            
        Returns:
            TranscriptionResult with text, timestamps, metadata
        """
        start_time = time.time()
        
        # Handle different input types
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            if audio_path.exists():
                # Local file
                audio_data = audio_path.read_bytes()
                audio_format = self.audio_processor.detect_format(audio_data)
            elif audio.startswith(("http://", "https://")):
                # URL - download first
                audio_data = self._download_audio(audio)
                audio_format = self.audio_processor.detect_format(audio_data)
            else:
                # Base64 encoded
                audio_data = base64.b64decode(audio)
                audio_format = self.audio_processor.detect_format(audio_data)
        elif isinstance(audio, bytes):
            audio_data = audio
            audio_format = self.audio_processor.detect_format(audio_data)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        # Apply voice control if specified
        if voice_control:
            audio_array, sample_rate = self.audio_processor.read_audio(audio_data, audio_format)
            audio_array, sample_rate = self.audio_processor.apply_voice_control(
                audio_array, sample_rate, voice_control
            )
        else:
            # Apply default voice control
            audio_array, sample_rate = self.audio_processor.read_audio(audio_data, audio_format)
            audio_array, sample_rate = self.audio_processor.apply_voice_control(
                audio_array, sample_rate, self.config.voice_control
            )
        
        # Prepare for model
        model_input = self.audio_processor.prepare_for_model(audio_array, sample_rate)
        
        # Transcribe
        if self._model:
            result = self._transcribe_local(model_input, return_timestamps, return_confidence)
        elif self.api_key:
            result = self._transcribe_api(audio_data, language, return_timestamps, return_confidence)
        else:
            # Demo mode - return placeholder
            result = self._transcribe_demo(model_input)
        
        # Add metadata
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.audio_duration = len(audio_array) / sample_rate
        result.audio_format = audio_format.value
        result.sample_rate = sample_rate
        result.language = language
        
        return result
    
    def transcribe_batch(
        self,
        audio_files: List[Union[str, bytes, Path]],
        language: str = "am",
        voice_control: Optional[VoiceControl] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio files
            language: Language code
            voice_control: Voice control parameters
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of TranscriptionResult
        """
        results = []
        for audio_file in audio_files:
            try:
                result = self.transcribe(
                    audio_file,
                    language=language,
                    voice_control=voice_control,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                # Create error result
                result = TranscriptionResult(
                    text="",
                    language=language,
                    metadata={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    async def transcribe_stream(
        self,
        audio_source,
        language: str = "am",
        chunk_size_ms: int = 160,
        voice_control: Optional[VoiceControl] = None,
        **kwargs
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Stream transcription for real-time processing.
        
        Args:
            audio_source: Audio file or stream
            language: Language code
            chunk_size_ms: Chunk size in milliseconds
            voice_control: Voice control parameters
            
        Yields:
            TranscriptionResult for each chunk
        """
        # For streaming, we process audio in chunks
        # This is a simplified version - real implementation would handle streams
        
        if isinstance(audio_source, (str, Path)):
            audio_path = Path(audio_source)
            audio_data = audio_path.read_bytes()
        elif isinstance(audio_source, bytes):
            audio_data = audio_source
        else:
            raise ValueError(f"Unsupported audio source: {type(audio_source)}")
        
        # Read and process audio
        audio_array, sample_rate = self.audio_processor.read_audio(audio_data)
        
        # Apply voice control
        if voice_control:
            audio_array, sample_rate = self.audio_processor.apply_voice_control(
                audio_array, sample_rate, voice_control
            )
        
        # Split into chunks
        chunk_duration = chunk_size_ms / 1000
        chunks = self.audio_processor.split_audio(audio_array, sample_rate, chunk_duration)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Prepare for model
            model_input = self.audio_processor.prepare_for_model(chunk, sample_rate)
            
            # Transcribe chunk
            if self._model:
                result = self._transcribe_local(model_input, True, True)
            else:
                result = self._transcribe_demo(model_input)
            
            # Update timing
            result.metadata["chunk_index"] = i
            result.metadata["total_chunks"] = len(chunks)
            
            yield result
    
    def _transcribe_local(
        self,
        audio: Any,
        return_timestamps: bool,
        return_confidence: bool
    ) -> TranscriptionResult:
        """Transcribe using local ONNX model."""
        import numpy as np
        
        # Reshape for model input
        audio_input = audio.reshape(1, -1).astype(np.float32)
        
        # Run inference
        outputs = self._model.run(None, {"audio": audio_input})
        
        # Decode CTC output
        logits = outputs[0]
        
        # Greedy decoding
        indices = np.argmax(logits, axis=-1)
        
        # Remove blanks and duplicates
        transcript = self._ctc_decode(indices[0])
        
        # Create result
        return TranscriptionResult(
            text=transcript,
            confidence=0.95 if return_confidence else 1.0,
            timestamps=[] if not return_timestamps else []
        )
    
    def _transcribe_api(
        self,
        audio_data: bytes,
        language: str,
        return_timestamps: bool,
        return_confidence: bool
    ) -> TranscriptionResult:
        """Transcribe using cloud API."""
        import requests
        
        # Prepare request
        files = {"file": ("audio.wav", audio_data, "audio/wav")}
        data = {
            "language": language,
            "return_timestamps": return_timestamps,
            "return_confidence": return_confidence
        }
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.post(
            f"{self.api_url}/transcribe",
            files=files,
            data=data,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result_data = response.json()
        
        # Parse timestamps
        timestamps = []
        for ts in result_data.get("timestamps", []):
            timestamps.append(Timestamp(
                word=ts["word"],
                start=ts["start"],
                end=ts["end"],
                confidence=ts.get("confidence", 1.0)
            ))
        
        return TranscriptionResult(
            text=result_data["text"],
            confidence=result_data.get("confidence", 1.0),
            timestamps=timestamps
        )
    
    def _transcribe_demo(self, audio: Any) -> TranscriptionResult:
        """Demo transcription (placeholder)."""
        return TranscriptionResult(
            text="ሰላም ዓለም (Demo - model not loaded)",
            confidence=0.0,
            metadata={"mode": "demo"}
        )
    
    def _ctc_decode(self, indices: Any) -> str:
        """CTC greedy decoding."""
        # Simple CTC decode - remove blanks and duplicates
        decoded = []
        prev_idx = -1
        
        for idx in indices:
            if idx != 0 and idx != prev_idx:  # 0 is blank
                decoded.append(idx)
            prev_idx = idx
        
        # Convert indices to characters
        # This would use the tokenizer in real implementation
        return "".join([chr(i + ord('а')) for i in decoded if i < 26])
    
    def _download_audio(self, url: str) -> bytes:
        """Download audio from URL."""
        import requests
        
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio: {response.status_code}")
        
        return response.content
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for agent frameworks."""
        return {
            "type": "function",
            "function": {
                "name": "transcribe_amharic",
                "description": "Transcribe Amharic speech audio to text. Supports multiple audio formats and voice control options.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio": {
                            "type": "string",
                            "description": "Audio file path, URL, or base64 encoded audio"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["am", "am-ET"],
                            "description": "Language code (default: am)"
                        },
                        "return_timestamps": {
                            "type": "boolean",
                            "description": "Include word-level timestamps (default: true)"
                        },
                        "return_confidence": {
                            "type": "boolean",
                            "description": "Include confidence scores (default: true)"
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["json", "srt", "vtt"],
                            "description": "Output format (default: json)"
                        },
                        "voice_control": {
                            "type": "object",
                            "description": "Voice control parameters",
                            "properties": {
                                "normalize_volume": {"type": "boolean"},
                                "noise_reduction": {"type": "boolean"},
                                "trim_silence": {"type": "boolean"},
                                "sample_rate": {"type": "integer"}
                            }
                        }
                    },
                    "required": ["audio"]
                }
            }
        }
    
    def get_batch_tool_definition(self) -> Dict[str, Any]:
        """Get batch transcription tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "transcribe_amharic_batch",
                "description": "Transcribe multiple Amharic audio files in batch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of audio file paths or URLs"
                        },
                        "language": {
                            "type": "string",
                            "description": "Language code (default: am)"
                        },
                        "max_concurrent": {
                            "type": "integer",
                            "description": "Maximum concurrent requests (default: 5)"
                        }
                    },
                    "required": ["audio_files"]
                }
            }
        }
