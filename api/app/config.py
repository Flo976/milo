from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    # Auth
    api_key: str = "change-me-in-production"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Claude API
    anthropic_api_key: str = ""
    claude_model: str = "claude-3-5-haiku-20241022"
    claude_timeout_s: float = 2.0

    # Local LLM
    local_llm_model_path: str = "/home/florent/milo/models/mistral-7b-instruct-v0.3-q4_k_m.gguf"
    local_llm_n_gpu_layers: int = -1

    # Models — local path or HuggingFace ID
    whisper_model_id: str = "/home/florent/milo/models/whisper-mg-v1/checkpoint-7000"
    nllb_model_id: str = "facebook/nllb-200-distilled-600M"
    mms_tts_model_id: str = "facebook/mms-tts-mlg"

    # Mode
    default_mode: str = "cloud"

    # Sessions
    session_ttl_s: int = 1800
    session_max_history: int = 10

    # Rate limiting
    rate_limit_free: int = 100
    rate_limit_pro: int = 10000

    # Monitoring
    prometheus_enabled: bool = True

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3001",
        "http://localhost:80",
        "https://milo.sooatek.com",
    ]

    # Limits
    stt_max_duration_s: int = 30
    tts_max_chars: int = 500
    upload_max_bytes: int = 10 * 1024 * 1024  # 10 MB

    model_config = {
        "env_file": ["../.env", ".env"],
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
