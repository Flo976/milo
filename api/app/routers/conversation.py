import asyncio
import json
import logging
import base64

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import settings
from app.dependencies import get_session_manager
from app.middleware.metrics import ACTIVE_WEBSOCKETS
from app.services.audio import audio_to_base64_wav
from app.services.pipeline import run_voice_pipeline

logger = logging.getLogger("milo")
router = APIRouter()

CHUNK_SAMPLES = int(16000 * 0.320)  # 320ms at 16kHz
MIN_SPEECH_SAMPLES = 16000 * 0.5  # At least 0.5s of audio to process


class AudioBuffer:
    """Accumulates PCM chunks and detects end-of-speech via silence."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.silence_frames = 0
        self.speech_started = False

    def add_chunk(self, pcm: np.ndarray, is_speech: bool) -> bool:
        """Add a chunk. Returns True when utterance is complete."""
        if is_speech:
            self.speech_started = True
            self.silence_frames = 0
            self.chunks.append(pcm)
        elif self.speech_started:
            self.silence_frames += 1
            self.chunks.append(pcm)
            # ~640ms of silence = end of utterance
            if self.silence_frames >= 2:
                return True
        return False

    @property
    def has_speech(self) -> bool:
        return self.speech_started and len(self.chunks) > 0

    def get_audio(self) -> np.ndarray:
        if not self.chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.chunks)

    def reset(self):
        self.chunks.clear()
        self.silence_frames = 0
        self.speech_started = False


async def process_utterance(ws: WebSocket, audio: np.ndarray, session_mgr, session_id: str):
    """Process a complete utterance through the pipeline and send results."""
    if len(audio) < MIN_SPEECH_SAMPLES:
        logger.info("Audio too short (%d samples), skipping", len(audio))
        return

    logger.info("Processing utterance: %d samples (%.1fs)", len(audio), len(audio) / 16000)

    # Get session history
    history = await session_mgr.get_history(session_id)

    # Run full pipeline
    result = await run_voice_pipeline(audio, history)

    # Send transcript
    await ws.send_json({
        "type": "transcript",
        "text": result["user_text_mg"],
        "partial": False,
    })

    # Send reply text
    await ws.send_json({
        "type": "reply_text",
        "text": result["reply_mg"],
    })

    # Send reply audio (TTS returns WAV bytes)
    if isinstance(result["audio"], bytes):
        audio_b64 = base64.b64encode(result["audio"]).decode("ascii")
    else:
        audio_b64 = audio_to_base64_wav(result["audio"])
    await ws.send_json({
        "type": "reply_audio",
        "audio": audio_b64,
    })

    # Send mode
    await ws.send_json({
        "type": "mode",
        "value": result["mode"],
    })

    # Store in session
    await session_mgr.add_exchange(
        session_id,
        result["user_text_mg"],
        result["reply_mg"],
    )


@router.websocket("/conversation")
async def conversation_ws(
    ws: WebSocket,
    api_key: str = Query(...),
    session_id: str = Query(default=""),
):
    # Auth
    if api_key != settings.api_key:
        await ws.close(code=4001, reason="Invalid API key")
        return

    await ws.accept()
    ACTIVE_WEBSOCKETS.inc()

    session_mgr = await get_session_manager()
    if not session_id or not await session_mgr.exists(session_id):
        session_id = await session_mgr.create()

    await ws.send_json({"type": "session", "session_id": session_id})

    # Get VAD (preloaded at startup)
    from app.models.model_manager import model_manager
    vad = model_manager.get("vad")
    if vad is None:
        from app.models.vad import load_vad
        vad = await asyncio.to_thread(load_vad)

    audio_buf = AudioBuffer()
    was_speaking = False
    chunk_count = 0

    try:
        while True:
            data = await ws.receive()

            # Log raw message type for debug
            msg_types = [k for k in data.keys() if k != "type"]
            logger.info("[WS-DBG] recv keys=%s", msg_types)

            # Text message (control commands)
            if "text" in data:
                msg = json.loads(data["text"])
                logger.info("[WS-DBG] TEXT msg: %s", msg)
                if msg.get("type") == "stop":
                    logger.info("[WS-DBG] STOP received. buf.has_speech=%s, buf.chunks=%d, buf.speech_started=%s",
                                audio_buf.has_speech, len(audio_buf.chunks), audio_buf.speech_started)
                    # Process whatever audio we have when user stops recording
                    if audio_buf.has_speech:
                        audio = audio_buf.get_audio()
                        logger.info("[WS-DBG] Processing buffered audio: %d samples (%.1fs), rms=%.4f",
                                    len(audio), len(audio) / 16000,
                                    float(np.sqrt(np.mean(audio**2))))
                        audio_buf.reset()
                        was_speaking = False
                        await ws.send_json({"type": "vad", "speaking": False})
                        await process_utterance(ws, audio, session_mgr, session_id)
                    else:
                        logger.info("[WS-DBG] STOP but no speech in buffer, resetting")
                        audio_buf.reset()
                    continue

            # Binary message (PCM int16 LE audio from Web Audio API)
            if "bytes" in data:
                chunk_count += 1
                pcm = np.frombuffer(data["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(pcm**2)))

                is_speech = await asyncio.to_thread(vad.is_speech, pcm)

                if chunk_count <= 5 or chunk_count % 30 == 0:
                    logger.info("[WS-DBG] chunk #%d: %d bytes, %d samples, rms=%.4f, vad=%s, buf_chunks=%d, speech_started=%s",
                                chunk_count, len(data["bytes"]), len(pcm), rms, is_speech,
                                len(audio_buf.chunks), audio_buf.speech_started)

                # Send VAD events
                if is_speech and not was_speaking:
                    logger.info("[WS-DBG] VAD: speech START")
                    await ws.send_json({"type": "vad", "speaking": True})
                elif not is_speech and was_speaking and not audio_buf.speech_started:
                    logger.info("[WS-DBG] VAD: speech END")
                    await ws.send_json({"type": "vad", "speaking": False})
                was_speaking = is_speech

                utterance_complete = audio_buf.add_chunk(pcm, is_speech)

                if utterance_complete:
                    audio = audio_buf.get_audio()
                    audio_buf.reset()
                    was_speaking = False
                    await ws.send_json({"type": "vad", "speaking": False})
                    await process_utterance(ws, audio, session_mgr, session_id)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", session_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        ACTIVE_WEBSOCKETS.dec()
