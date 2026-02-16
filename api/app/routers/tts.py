import base64
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.config import settings
from app.schemas.tts import TTSRequest, TTSResponse
from app.services.pipeline import run_tts

logger = logging.getLogger("milo")
router = APIRouter()


@router.post("/tts", summary="Text-to-Speech")
async def text_to_speech(req: TTSRequest):
    """Synthetise du texte malagasy en audio avec MMS-TTS.

    - `format=wav` : retourne directement le fichier WAV (binary)
    - `format=json` : retourne l'audio en base64 dans du JSON

    Max 500 caracteres.
    """
    if len(req.text) > settings.tts_max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds {settings.tts_max_chars} character limit",
        )

    audio_bytes, processing_ms = await run_tts(req.text, req.language)

    if req.format == "wav":
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Processing-Ms": str(round(processing_ms)),
            },
        )

    # JSON response with base64
    return TTSResponse(
        audio=base64.b64encode(audio_bytes).decode("ascii"),
        format=req.format,
        sample_rate=16000,
        processing_ms=round(processing_ms),
    )
