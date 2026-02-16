from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str | None = Field(default=None, description="Session UUID")
    language: str = Field(default="mg", description="Language code")
    mode: str = Field(default="text", description="text or voice")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    mode: str = Field(description="local or cloud")
    processing_ms: float
