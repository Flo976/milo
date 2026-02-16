import base64
import logging

from fastapi import APIRouter, HTTPException

from app.dependencies import get_session_manager
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.pipeline import run_llm, run_tts

logger = logging.getLogger("milo")
router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Chat texte avec Milo")
async def chat(req: ChatRequest):
    """Envoie un message texte et recoit une reponse de Milo.

    Le LLM (Claude ou Mistral local) repond directement en malagasy.
    Les sessions conservent l'historique de conversation.

    Avec `mode=voice`, la reponse inclut l'audio TTS en base64.

    **Exemple :** `{"message": "Salama, iza ianao?"}`
    """
    if not req.message or not req.message.strip():
        raise HTTPException(
            status_code=422,
            detail="Message must not be empty",
        )

    session_mgr = await get_session_manager()

    # Get or create session
    session_id = req.session_id
    if session_id is None or not await session_mgr.exists(session_id):
        session_id = await session_mgr.create()

    history = await session_mgr.get_history(session_id)

    # LLM responds directly in malagasy (no NLLB translation needed)
    reply, mode, processing_ms = await run_llm(req.message, history)

    # Generate TTS audio if voice mode requested
    audio_b64 = None
    if req.mode == "voice":
        audio_bytes, tts_ms = await run_tts(reply)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        processing_ms += tts_ms

    # Store exchange in session
    await session_mgr.add_exchange(session_id, req.message, reply)

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        mode=mode,
        processing_ms=round(processing_ms),
        audio=audio_b64,
    )
