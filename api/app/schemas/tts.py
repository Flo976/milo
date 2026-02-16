from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=500, description="Text to synthesize")
    language: str = Field(default="mg", description="Language code")
    format: str = Field(default="wav", description="Output audio format")


class TTSResponse(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio")
    format: str = "wav"
    sample_rate: int = 16000
    processing_ms: float
