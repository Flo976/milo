from pydantic import BaseModel


class ConversationEvent(BaseModel):
    type: str
    speaking: bool | None = None
    text: str | None = None
    partial: bool | None = None
    audio: str | None = None
    value: str | None = None
    error: str | None = None
