from pydantic import BaseModel, Field


class STTRequest(BaseModel):
    audio: str = Field(..., description="Base64-encoded WAV audio")
    format: str = Field(default="wav", description="Audio format")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")


class STTResponse(BaseModel):
    text: str
    language: str = "mg"
    confidence: float = Field(ge=0.0, le=1.0)
    duration_ms: float
    processing_ms: float
