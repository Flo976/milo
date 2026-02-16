from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str
    models_loaded: list[str]
    vram: dict
    redis_connected: bool
