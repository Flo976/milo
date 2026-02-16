from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source: str = Field(default="mg", description="Source language (mg or fr)")
    target: str = Field(default="fr", description="Target language (mg or fr)")


class TranslateResponse(BaseModel):
    translation: str
    source: str
    target: str
    processing_ms: float
