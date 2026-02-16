import asyncio
import logging
import time

from app.middleware.metrics import (
    LLM_FALLBACK,
    LLM_REQUESTS,
    PIPELINE_STAGE_DURATION,
)
from app.models.model_manager import model_manager
from app.services.fallback import fallback_manager

logger = logging.getLogger("milo")


async def run_stt(audio_float32) -> tuple[str, float, float]:
    """Run STT on audio. Returns (text, confidence, processing_ms)."""
    from app.models.stt import load_stt
    stt_model = await model_manager.get_or_load("stt", load_stt)

    start = time.perf_counter()
    async with model_manager.gpu_semaphore:
        text, confidence = await asyncio.to_thread(stt_model.transcribe, audio_float32)
    elapsed = (time.perf_counter() - start) * 1000
    return text, confidence, elapsed


async def run_tts(text: str, language: str = "mg") -> tuple[bytes, float]:
    """Run TTS on text. Returns (audio_bytes_wav, processing_ms)."""
    from app.models.tts import load_tts
    tts_model = await model_manager.get_or_load("tts", load_tts)

    start = time.perf_counter()
    async with model_manager.gpu_semaphore:
        audio = await asyncio.to_thread(tts_model.synthesize, text)
    elapsed = (time.perf_counter() - start) * 1000
    return audio, elapsed


async def run_translate(
    text: str, source: str, target: str
) -> tuple[str, float]:
    """Translate text. Returns (translated, processing_ms)."""
    from app.models.translator import load_translator
    translator = await model_manager.get_or_load("translator", load_translator)

    start = time.perf_counter()
    async with model_manager.gpu_semaphore:
        result = await asyncio.to_thread(
            translator.translate, text, source, target
        )
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


async def run_llm(
    message: str, history: list[dict]
) -> tuple[str, str, float]:
    """Run LLM. Returns (reply, mode_used, processing_ms)."""
    start = time.perf_counter()

    # Try cloud LLM
    if fallback_manager.should_try_cloud():
        try:
            from app.models.llm_cloud import load_llm_cloud
            llm_cloud = await model_manager.get_or_load("llm_cloud", load_llm_cloud)

            reply = await llm_cloud.generate(message, history)
            fallback_manager.report_cloud_success()
            elapsed = (time.perf_counter() - start) * 1000
            LLM_REQUESTS.labels(mode="cloud").inc()
            return reply, "cloud", elapsed
        except Exception as e:
            logger.warning("Cloud LLM failed: %s, falling back to local", e)
            fallback_manager.report_cloud_failure()
            LLM_FALLBACK.inc()

    # Try local LLM
    try:
        from app.models.llm_local import load_llm_local
        llm_local = await model_manager.get_or_load("llm_local", load_llm_local)

        async with model_manager.gpu_semaphore:
            reply = await asyncio.to_thread(llm_local.generate, message, history)
        elapsed = (time.perf_counter() - start) * 1000
        LLM_REQUESTS.labels(mode="local").inc()
        return reply, "local", elapsed
    except Exception as e:
        logger.warning("Local LLM unavailable: %s, using echo mode", e)

    # No LLM available — echo mode
    elapsed = (time.perf_counter() - start) * 1000
    LLM_REQUESTS.labels(mode="echo").inc()
    return f"[Echo] {message}", "echo", elapsed


async def run_voice_pipeline(
    audio_float32, session_history: list[dict]
) -> dict:
    """Voice pipeline: STT MG -> LLM (responds in MG directly) -> TTS MG."""
    timings = {}

    # STT
    import numpy as np
    rms = float(np.sqrt(np.mean(audio_float32**2)))
    logger.info("[PIPELINE] STT input: %d samples (%.1fs), dtype=%s, rms=%.4f, min=%.3f, max=%.3f",
                len(audio_float32), len(audio_float32) / 16000, audio_float32.dtype,
                rms, float(audio_float32.min()), float(audio_float32.max()))
    text_mg, stt_confidence, t = await run_stt(audio_float32)
    timings["stt_ms"] = round(t)
    PIPELINE_STAGE_DURATION.labels(stage="stt").observe(t / 1000)
    logger.info("[PIPELINE] STT output: '%s' (confidence=%.3f, %d chars, %dms)", text_mg, stt_confidence, len(text_mg), round(t))

    # LLM (responds directly in malagasy, no translation needed)
    reply_mg, mode, t = await run_llm(text_mg, session_history)
    timings["llm_ms"] = round(t)
    PIPELINE_STAGE_DURATION.labels(stage="llm").observe(t / 1000)
    logger.info("[PIPELINE] LLM (%s): '%s' -> '%s' (%dms)", mode, text_mg, reply_mg, round(t))

    # TTS
    audio_bytes, t = await run_tts(reply_mg)
    timings["tts_ms"] = round(t)
    PIPELINE_STAGE_DURATION.labels(stage="tts").observe(t / 1000)
    logger.info("[PIPELINE] TTS: %d bytes (%dms)", len(audio_bytes) if audio_bytes else 0, round(t))

    timings["total_ms"] = sum(timings.values())

    return {
        "user_text_mg": text_mg,
        "reply_mg": reply_mg,
        "audio": audio_bytes,
        "mode": mode,
        "timings": timings,
    }
