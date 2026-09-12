# Amharic ASR

State-of-the-art Amharic Automatic Speech Recognition system trained from scratch.

## Features

- **27M parameter model** - optimized for edge deployment
- **Multi-format support** - WAV, MP3, FLAC, OGG, M4A, WebM, and more
- **Voice control** - pitch, speed, volume, normalization, noise reduction
- **Agent-ready** - tool calling for LangChain, CrewAI, AutoGPT
- **Streaming** - real-time transcription with bounded latency
- **INT4 quantized** - only 4MB model size

## Quick Start

```python
from amharic_asr import AmharicASR

client = AmharicASR()
result = client.transcribe("audio.wav")
print(result.text)
```

## Installation

```bash
pip install amharic-asr-agent
```

## License

MIT
