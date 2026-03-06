# Voice Integration Plan

## Speech-to-Text (STT)

STT transcription has no content filtering issues.

| Option | Runs on | Notes |
|--------|---------|-------|
| **Faster-Whisper** | Server | Best speed/accuracy balance. CTranslate2 backend, ~10x faster than vanilla Whisper. Python lib. |
| **Whisper.cpp** | Server | C++ port, very low latency. Good as a sidecar process. |
| **Web Speech API** | Browser | Free, zero setup, inconsistent across browsers, Chrome sends audio to Google. |
| **Whisper via LM Studio** | Existing GPU server | LM Studio supports Whisper models alongside LLMs. |

**Recommendation:** Faster-Whisper on the backend. Expose `POST /v1/transcribe` endpoint. Use `whisper-large-v3-turbo` for quality/speed balance.

## Text-to-Speech (TTS) — NSFW-compatible

Most cloud APIs (ElevenLabs, OpenAI, Google) refuse or filter explicit content. Local options have no restrictions.

| Option | Quality | Speed | Voice cloning | Notes |
|--------|---------|-------|---------------|-------|
| **Kokoro** | Very good | Fast | No (preset voices) | 82M params, runs on CPU. Natural prosody. Apache 2.0. |
| **Piper** | Good | Very fast | No (pretrained) | Runs on Raspberry Pi-level hardware. Great latency. |
| **XTTS v2** | Excellent | Moderate | Yes (few-shot) | Clone any voice from ~6s sample. ~1.5GB VRAM. |
| **GPT-SoVITS** | Excellent | Moderate | Yes | Popular in companion space. Good for anime/character voices. |
| **Fish Speech** | Very good | Fast | Yes | Newer, clean API. Has both local and permissive hosted API. |
| **AllTalk TTS** | Varies | Varies | Depends | Wrapper with web UI — swap between Piper, XTTS, etc. |

**Recommendation:**
- **Kokoro** for fast, simple, low resource usage
- **XTTS v2** for voice cloning (give the companion a custom voice from a sample)

## Architecture

```
Browser (mic) --> POST audio --> Backend --> Faster-Whisper --> text
                                                                 |
                                                            /v1/chat
                                                                 |
text <-- <audio> element <-- Backend streams audio <-- TTS <-- response
```

Two new FastAPI endpoints:
- `POST /v1/transcribe` — accepts audio blob, returns text
- `POST /v1/synthesize` — accepts text, streams back audio (or add `voice=true` param to `/v1/chat`)

Browser side needs `MediaRecorder` for capture and `<audio>` element for playback — no special libraries.

## Implementation Order

1. Faster-Whisper STT backend endpoint (simpler integration)
2. TTS engine selection and backend endpoint
3. Web UI: mic button + audio playback
4. Streaming: wire TTS into SSE chat flow for lower latency
