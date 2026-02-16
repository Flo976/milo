import logging
import time

from fastapi import APIRouter, HTTPException

from app.schemas.stt import STTRequest, STTResponse
from app.services.audio import (
    AudioValidationError,
    decode_base64_audio,
    ensure_mono_16k,
    validate_duration,
)
from app.services.pipeline import run_stt

logger = logging.getLogger("milo")
router = APIRouter()


@router.post("/stt", response_model=STTResponse, summary="Speech-to-Text")
async def speech_to_text(req: STTRequest):
    """Transcrit un audio en texte malagasy avec Whisper fine-tune.

    L'audio doit etre encode en base64 (format WAV, 16kHz mono).
    Duree max : 30 secondes.
    """
    try:
        audio, sr = decode_base64_audio(req.audio)
        audio = ensure_mono_16k(audio, sr)
        duration = validate_duration(audio)
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    text, confidence, processing_ms = await run_stt(audio)

    return STTResponse(
        text=text,
        language="mg",
        confidence=round(confidence, 4),
        duration_ms=round(duration * 1000),
        processing_ms=round(processing_ms),
    )
