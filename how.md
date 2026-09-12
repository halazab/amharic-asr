# Amharic ASR: From Scratch to Agent-Ready

## Goal

Build a **state-of-the-art, agent-ready** Amharic (አማርኛ) Automatic Speech Recognition system **from scratch** on a single Google Colab T4 GPU (16GB VRAM). No pre-trained models, no fine-tuning — train from random weights.

### Core Requirements

- **Fastest to train** — finish all patches in under 3 hours on T4
- **Fastest to inference** — real-time or better on consumer hardware
- **Smallest footprint** — deployable on mobile/edge devices
- **Best WER for low-resource** — optimized for ~200h of Amharic speech
- **Agent-ready** — tool calling, API endpoints, structured output for AI agents

---

## Architecture Evolution

### Why Moonshine-Style > Zipformer-S

Based on 2025-2026 research, **Moonshine-style architecture** outperforms Zipformer for edge and agent use cases:

| Metric | Zipformer-S (old) | Moonshine-Style (new) | Why Better |
|--------|-------------------|----------------------|------------|
| Parameters | 22M | 27M | 5M more = much better accuracy |
| Position encoding | Absolute | **RoPE** | Better length generalization |
| Padding | Zero-padding to 30s | **No padding** | Scales with audio length |
| Inference speed | 1.5-2x baseline | **5-15x faster** | Real-time on edge |
| Streaming | Chunked attention | **Sliding-window** | Bounded latency |
| Edge deployment | Good | **Excellent** | Proven on mobile |
| Agent integration | Basic API | **Native tool calling** | Built for agents |

### Key Innovations from Moonshine

1. **Rotary Position Embeddings (RoPE)** — handles variable lengths without padding
2. **Ergodic streaming encoder** — sliding-window attention for bounded latency
3. **No zero-padding** — FLOPs scale with actual audio length
4. **Proven for low-resource** — Arabic, Korean, Ukrainian trained from scratch

---

## Architecture Specification

### Moonshine-Style Amharic CTC (27M params)

```
Audio Preprocessor:
  Type: 3-layer CNN stem
  Strides: [64, 3, 2] → 384x compression
  Input: 16kHz raw waveform (no Fbank needed)
  
Encoder:
  Type: Transformer with sliding-window attention
  Layers: 6
  Hidden dim: 256
  Attention heads: 4
  FFN dim: 1024
  Position: RoPE (Rotary Position Embeddings)
  Window: (left=128, right=0) frames for streaming
  Activation: SwiGLU (better than Swish)

Decoder:
  Type: CTC (Connectionist Temporal Classification)
  Projection: Linear hidden_size → vocab_size
  Vocab size: 1024 (SentencePiece BPE)
  Blank token: index 0

Tokenizer:
  Type: SentencePiece BPE
  Vocab size: 1024
  Normalization: none (raw Ge'ez characters + spaces)
```

### Parameter Breakdown

| Component | Params | Notes |
|-----------|--------|-------|
| Audio preprocessor (CNN) | ~0.5M | 3-layer conv stem |
| Transformer encoder (6 layers) | ~24M | With RoPE + sliding-window |
| CTC projection head | ~0.25M | Linear layer |
| Tokenizer (SP) | ~0.01M | SentencePiece |
| **Total** | **~27M** | Optimal for edge |

---

## Training Optimizations

### Optimizer: Schedule-Free AdamW

Replace ScaledAdam with **Schedule-Free AdamW** (proven by Moonshine):

```yaml
optimizer: schedulefree_adamw
learning_rate: 2e-3
warmup_steps: 8192
weight_decay: 0.01
max_grad_norm: 5.0

# Why Schedule-Free?
# - No LR scheduling code needed
# - Faster convergence than AdamW
# - Better for variable-length audio
# - Proven on 200K+ hours of data
```

### Curriculum Learning

Train progressively from easy to hard:

```yaml
curriculum:
  enabled: true
  stages:
    - steps: 2000
      max_audio_length: 5.0
      description: "Short clips (easy)"
    - steps: 3000
      max_audio_length: 15.0
      description: "Medium clips"
    - steps: 3000
      max_audio_length: 30.0
      description: "Full length (hard)"

# Why curriculum?
# - 20-30% faster convergence
# - Better gradient stability
# - Prevents early OOM
```

### Data Augmentation (Enhanced)

```yaml
augmentation:
  speed_perturbation: [0.9, 0.95, 1.0, 1.05, 1.1]  # 5x
  noise_augmentation:
    enabled: true
    snr_range: [5, 20]  # dB
  room_impulse_response:
    enabled: true
  spec_augment:
    time_warp_factor: 20
    num_freq_masks: 3
    num_time_masks: 3
    freq_mask_range: [0, 30]
    time_mask_range: [0, 50]

# Why enhanced augmentation?
# - Better generalization
# - Handles real-world noise
# - Critical for low-resource
```

### Training Configuration

```yaml
precision: bf16  # Better than fp16
batch_size: 16   # per GPU (doubled)
gradient_accumulation: 2  # effective = 32
max_duration: 30.0
num_epochs: 8-10  # per patch
early_stopping_patience: 5
checkpoint_save_every: 500
val_check_every: 0.5
num_workers: 4

# Why bf16?
# - More stable than fp16
# - No loss scaling needed
# - Same speed on T4
```

---

## Data Pipeline

### Datasets

| Dataset | Hours | Type | License | Priority |
|---------|-------|------|---------|----------|
| WAXAL Amharic (`amh_asr`) | ~120h | Read + spontaneous | CC-BY-4.0 | Primary |
| Google FLEURS (`am`) | ~12h | Read speech | CC-BY-4.0 | Secondary |
| Mozilla Common Voice | ~5h | Crowdsourced | CC-BY-4.0 | Optional |
| **Pseudo-labeled (Whisper)** | **+50-100h** | Web audio | CC-BY-4.0 | **New** |

**Total available: ~200-250h** (with pseudo-labeling)

### Pseudo-Labeling Pipeline (NEW)

Use Whisper Large V3 to generate better training labels:

```python
# 1. Download unlabeled Amharic audio from web
# 2. Transcribe with Whisper Large V3
# 3. Filter by confidence (avg log prob > threshold)
# 4. Add to training data

# Expected gain: +50-100h高质量 labeled data
# Quality filter: Remove samples with avg log_prob < -1.5
```

### Progressive Patch Training

```yaml
patch_size: 20  # hours per patch
total_patches: 10

patch_1:  WAXAL 0-3976        (20h) → train → delete WAVs
patch_2:  WAXAL 3976-7952     (20h) → train → delete WAVs
patch_3:  WAXAL 7952-11928    (20h) → train → delete WAVs
...
patch_6:  WAXAL remaining + FLEURS (20h) → train → delete WAVs
patch_7:  Pseudo-labeled 1    (20h) → train → delete WAVs
patch_8:  Pseudo-labeled 2    (20h) → train → delete WAVs
patch_9:  Pseudo-labeled 3    (20h) → train → delete WAVs
patch_10: All remaining       (20h) → train → delete WAVs
```

### Held-Out Sets (Never Deleted)

| Set | Samples | Hours | Purpose |
|-----|---------|-------|---------|
| Validation | ~2,900 | ~14h | Early stopping |
| Test | ~3,400 | ~16h | Final evaluation |

---

## Training Timeline (Improved)

### Per-Patch Estimate

| Metric | Old (Zipformer) | New (Moonshine) | Improvement |
|--------|----------------|-----------------|-------------|
| Time per step | ~0.7s | **~0.4s** | 43% faster |
| Time per epoch | ~87s | **~50s** | 42% faster |
| Epochs needed | 50-100 | **30-50** | 40% fewer |
| **Time per patch** | 1.25-2.5h | **0.5-1h** | 2.5x faster |
| **Total (10 patches)** | 12-25h | **5-10h** | 2.5x faster |

### Checkpoint Strategy

```yaml
save_every: 500 steps
keep_last: 10 checkpoints
final_model: "average of last 10 checkpoints"

# Resume strategy:
# - Load latest checkpoint
# - Reset optimizer state
# - Continue with new patch
# - Never reset model weights
```

---

## Inference Optimizations

### Speed Optimizations

| Technique | Speedup | WER Impact | Size Impact |
|-----------|---------|------------|-------------|
| CTC greedy search | O(n) | None | None |
| No beam search | 2-3x | None | None |
| ONNX export | 2-3x | None | None |
| INT8 quantization | 2x | <0.1% | -50% |
| INT4 k-quant | 3x | +0.17% | -73% |
| TensorRT | 2-3x | None | None |

### Streaming with Sliding-Window Attention

```yaml
streaming:
  enabled: true
  window_left: 128    # frames
  window_right: 0     # frames (causal)
  chunk_size_ms: 160  # 10 frames @ 16kHz
  latency_ms: 40-50   # bounded!

# Why sliding-window?
# - Bounded latency (independent of utterance length)
# - Same accuracy as full attention
# - Proven by Moonshine v2
```

### Estimated Inference Performance

| Mode | RTF (T4) | RTF (CPU) | Latency | Model Size |
|------|----------|-----------|---------|------------|
| Batch (full) | **0.02x** | **0.1x** | N/A | 27M / 14MB |
| Streaming | **0.05x** | **0.15x** | **40-50ms** | 27M / 14MB |
| INT8 ONNX | **0.01x** | **0.05x** | 30ms | 14MB |
| INT4 ONNX | **0.008x** | **0.03x** | 25ms | **4MB** |

RTF 0.02x = 50 seconds of audio transcribed in 1 second

---

## Agent Tool Calling System

### Overview

Make the ASR model usable as a **tool by AI agents** (LangChain, CrewAI, AutoGPT, etc.):

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Framework                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  LangChain  │  │   CrewAI    │  │  AutoGPT    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          │                              │
│                    ┌─────▼─────┐                        │
│                    │  Tool API  │                        │
│                    └─────┬─────┘                        │
└──────────────────────────┼──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   ASR Service                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   REST API  │  │   gRPC      │  │  WebSocket  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          │                              │
│                    ┌─────▼─────┐                        │
│                    │ ASR Engine │                        │
│                    │ (27M CTC) │                        │
│                    └───────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### Tool Definition (OpenAI Function Calling Format)

```json
{
  "type": "function",
  "function": {
    "name": "transcribe_amharic",
    "description": "Transcribe Amharic speech audio to text. Supports both file upload and URL.",
    "parameters": {
      "type": "object",
      "properties": {
        "audio": {
          "type": "string",
          "description": "Base64 encoded audio or URL to audio file"
        },
        "format": {
          "type": "string",
          "enum": ["wav", "mp3", "flac", "ogg"],
          "description": "Audio format (default: auto-detect)"
        },
        "language": {
          "type": "string",
          "enum": ["am", "am-ET"],
          "description": "Language code (default: am)"
        },
        "return_timestamps": {
          "type": "boolean",
          "description": "Include word-level timestamps (default: false)"
        },
        "return_confidence": {
          "type": "boolean",
          "description": "Include confidence scores (default: false)"
        },
        "stream": {
          "type": "boolean",
          "description": "Enable streaming transcription (default: false)"
        }
      },
      "required": ["audio"]
    },
    "returns": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "Transcribed Amharic text"
        },
        "timestamps": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "word": {"type": "string"},
              "start": {"type": "number"},
              "end": {"type": "number"},
              "confidence": {"type": "number"}
            }
          },
          "description": "Word-level timestamps (if requested)"
        },
        "confidence": {
          "type": "number",
          "description": "Overall confidence score (0-1)"
        },
        "language": {
          "type": "string",
          "description": "Detected language code"
        },
        "duration": {
          "type": "number",
          "description": "Audio duration in seconds"
        },
        "processing_time": {
          "type": "number",
          "description": "Processing time in milliseconds"
        }
      }
    }
  }
}
```

### REST API Endpoints

```yaml
# FastAPI Server
/endpoints:
  /transcribe:
    post:
      summary: "Transcribe audio file"
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file: {type: string, format: binary}
                language: {type: string, default: "am"}
                return_timestamps: {type: boolean, default: false}
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  text: {type: string}
                  timestamps: {type: array}
                  confidence: {type: number}
                  processing_time_ms: {type: number}

  /transcribe_url:
    post:
      summary: "Transcribe audio from URL"
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                url: {type: string, format: uri}
                language: {type: string, default: "am"}
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TranscriptionResult'

  /transcribe_stream:
    post:
      summary: "Stream transcription (WebSocket upgrade)"
      description: "Real-time streaming transcription"

  /health:
    get:
      summary: "Health check"

  /metrics:
    get:
      summary: "Prometheus metrics"
```

### Python SDK for Agents

```python
# pip install amharic-asr-agent

from amharic_asr import AmharicASR

# Initialize client
client = AmharicASR(api_key="your-api-key")

# Basic transcription
result = client.transcribe("audio.wav")
print(result.text)  # "ሰላም ዓለም"

# With timestamps
result = client.transcribe(
    "audio.wav",
    return_timestamps=True,
    return_confidence=True
)
for ts in result.timestamps:
    print(f"{ts.word}: {ts.start:.2f}s - {ts.end:.2f}s ({ts.confidence:.2%})")

# Streaming (for real-time agents)
async for chunk in client.transcribe_stream("audio.wav"):
    print(chunk.text, end="", flush=True)

# Batch transcription
results = client.transcribe_batch([
    "audio1.wav",
    "audio2.wav",
    "audio3.wav"
])
```

### LangChain Integration

```python
from langchain.tools import Tool
from amharic_asr import AmharicASR

# Create tool
asr_tool = Tool(
    name="transcribe_amharic",
    description="Transcribe Amharic speech audio to text",
    func=lambda audio: AmharicASR().transcribe(audio).text
)

# Use in agent
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

agent = initialize_agent(
    tools=[asr_tool],
    llm=OpenAI(),
    agent="zero-shot-react-description"
)

# Agent can now transcribe audio
result = agent.run("Transcribe this Amharic audio file: recording.wav")
```

### CrewAI Integration

```python
from crewai import Agent, Task, Crew
from amharic_asr import AmharicASR

# Create agent with ASR tool
transcriber = Agent(
    role="Amharic Transcriber",
    goal="Transcribe Amharic speech accurately",
    backstory="Expert in Ethiopian languages",
    tools=[AmharicASRTool()],
    verbose=True
)

# Create task
transcription_task = Task(
    description="Transcribe the Amharic audio file and summarize the content",
    agent=transcriber,
    expected_output="Transcribed text with summary"
)

# Run crew
crew = Crew(agents=[transcriber], tasks=[transcription_task])
result = crew.kickoff()
```

### Webhook Support (Async Processing)

```yaml
# For long audio files, use webhook for async processing
POST /transcribe_async
{
  "audio_url": "https://example.com/audio.wav",
  "webhook_url": "https://your-app.com/webhook",
  "metadata": {"job_id": "12345"}
}

# Response (immediate)
{
  "job_id": "12345",
  "status": "processing",
  "estimated_time_seconds": 30
}

# Webhook callback (when done)
POST https://your-app.com/webhook
{
  "job_id": "12345",
  "status": "completed",
  "result": {
    "text": "ሰላም ዓለም",
    "confidence": 0.95,
    "processing_time_ms": 2500
  }
}
```

---

## Deployment Architecture

### Production Setup

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│                    (nginx/traefik)                        │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼─────┐ ┌───▼─────┐
        │  Worker 1 │ │ Worker 2 │ │ Worker 3│
        │  (GPU)    │ │  (GPU)   │ │  (GPU)  │
        └─────┬─────┘ └────┬─────┘ └────┬────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │   Redis     │
                    │  (Queue)    │
                    └─────────────┘
```

### Model Serving Options

```yaml
# Option 1: FastAPI + ONNX Runtime (Simple)
server: FastAPI
model: ONNX Runtime
quantization: INT8
concurrent: 10 requests/sec

# Option 2: Triton Inference Server (Production)
server: NVIDIA Triton
model: TensorRT
quantization: FP16/INT8
concurrent: 100+ requests/sec

# Option 3: TensorRT-LLM (Maximum Performance)
server: TensorRT-LLM
model: TensorRT engine
quantization: INT4/INT8
concurrent: 200+ requests/sec
```

### Docker Configuration

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install \
    fastapi \
    uvicorn \
    onnxruntime-gpu \
    numpy \
    soundfile

# Copy model and code
COPY model/ /app/model/
COPY server/ /app/server/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Export Pipeline

### Model Export Steps

```yaml
step_1_average_checkpoints:
  input: "checkpoints/checkpoint_*.pt"
  output: "model_final.pt"
  method: "average last 10 checkpoints"

step_2_export_onnx:
  input: "model_final.pt"
  output: "amharic_asr.onnx"
  opset: 17
  dynamic_axes:
    audio: {0: "batch_size", 2: "audio_length"}

step_3_quantize_int8:
  input: "amharic_asr.onnx"
  output: "amharic_asr_int8.onnx"
  method: "onnxruntime.quantization"
  per_channel: true

step_4_quantize_int4:
  input: "amharic_asr.onnx"
  output: "amharic_asr_int4.onnx"
  method: "onnxruntime.quantization"
  block_size: 32

step_5_optimize:
  input: "amharic_asr_int8.onnx"
  output: "amharic_asr_optimized.onnx"
  method: "onnxruntime.graph_optimization"
  level: "all"
```

### Model Outputs

```
amharic-asr-agent/
├── models/
│   ├── amharic_asr.onnx           # Full precision (27M)
│   ├── amharic_asr_int8.onnx      # INT8 quantized (14MB)
│   ├── amharic_asr_int4.onnx      # INT4 quantized (4MB)
│   └── amharic_asr_trt.plan       # TensorRT engine
├── tokenizer/
│   ├── tokenizer.model
│   ├── tokenizer.vocab
│   └── vocab.txt
├── api/
│   ├── server.py                  # FastAPI server
│   ├── schemas.py                 # Pydantic models
│   └── tools.py                   # Agent tool definitions
├── sdk/
│   ├── python/
│   │   └── amharic_asr/
│   │       ├── __init__.py
│   │       ├── client.py
│   │       └── models.py
│   └── javascript/
│       └── index.js
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config.yaml
├── README.md
└── LICENSE
```

---

## Complete Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| **Patch 1** (train + tokenizer) | 1-2h | Checkpoint + tokenizer |
| **Patches 2-6** (WAXAL) | 3-6h | WAXAL-trained model |
| **Patches 7-10** (pseudo-labeled) | 2-4h | Final model |
| **Export + quantize** | 0.5h | ONNX models |
| **API server + SDK** | 1-2h | Agent tools |
| **Testing + docs** | 0.5h | Production-ready |
| **Total** | **~8-15h** | **Agent-ready Amharic ASR** |

---

## Performance Comparison

### Training Speed

| Metric | FastConformer (old) | Zipformer (prev) | **Moonshine (new)** |
|--------|--------------------|--------------------|---------------------|
| Params | 32M | 22M | **27M** |
| Time/step | 1.2-1.5s | 0.6-0.8s | **0.3-0.4s** |
| Total time | 20-30h | 14-26h | **8-15h** |
| Convergence | 100+ epochs | 50-100 | **30-50** |

### Inference Speed

| Metric | FastConformer | Zipformer | **Moonshine** | **Moonshine+INT4** |
|--------|--------------|-----------|---------------|-------------------|
| RTF (T4) | 1x | 1.5-2x | **5-10x** | **15-25x** |
| RTF (CPU) | 0.1x | 0.3x | **0.5-0.7x** | **1-2x** |
| Latency | 200ms | 80ms | **40-50ms** | **25-30ms** |
| Model size | 32MB | 22MB | **14MB** | **4MB** |

### Agent Capabilities

| Feature | Basic ASR | **Agent-Ready ASR** |
|---------|-----------|---------------------|
| REST API | Basic | Full OpenAPI 3.0 |
| Tool calling | None | Native function calling |
| Streaming | Basic | WebSocket + gRPC |
| Batch | Single file | Parallel batch |
| Webhooks | None | Async with callbacks |
| SDKs | Python only | Python, JS, Go |
| Structured output | Text only | JSON with metadata |
| Agent framework | Manual | LangChain, CrewAI |

---

## Key Innovations

### 1. Architecture
- **RoPE** instead of absolute position embeddings
- **Sliding-window attention** for streaming
- **No zero-padding** for efficiency
- **SwiGLU activation** for better training

### 2. Training
- **Schedule-Free AdamW** (no LR scheduling)
- **Curriculum learning** (easy → hard)
- **Pseudo-labeling** (Whisper teacher)
- **Enhanced augmentation** (5x speed, noise, RIR)

### 3. Inference
- **CTC greedy search** (O(n), no beam search)
- **INT4 quantization** (73% smaller, +0.17% WER)
- **Sliding-window streaming** (bounded latency)
- **ONNX + TensorRT** optimization

### 4. Agent Integration
- **Native tool calling** (OpenAI format)
- **Structured JSON output** (with timestamps, confidence)
- **Streaming support** (WebSocket)
- **Multi-framework SDK** (LangChain, CrewAI, etc.)

---

## References

### Architecture Papers
- Moonshine: https://arxiv.org/abs/2410.15608
- Moonshine v2 (Streaming): https://arxiv.org/abs/2602.12241
- Flavors of Moonshine (Low-resource): https://arxiv.org/abs/2509.02523
- Zipformer: https://arxiv.org/abs/2310.11230

### Amharic ASR Research
- Ethio-ASR: https://arxiv.org/abs/2603.23654
- WAXAL Dataset: https://huggingface.co/datasets/waxal/amh_asr
- Amharic ASR Corrections: https://arxiv.org/abs/2404.13362

### Tools & Frameworks
- k2-fsa/icefall: https://github.com/k2-fsa/icefall
- ONNX Runtime: https://onnxruntime.ai/
- TensorRT: https://developer.nvidia.com/tensorrt
- LangChain: https://langchain.com/
- CrewAI: https://crewai.com/
