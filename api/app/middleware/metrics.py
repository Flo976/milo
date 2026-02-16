from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Custom metrics
REQUEST_COUNT = Counter(
    "milo_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

REQUEST_DURATION = Histogram(
    "milo_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
)

PIPELINE_STAGE_DURATION = Histogram(
    "milo_pipeline_stage_duration_seconds",
    "Pipeline stage duration",
    ["stage"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

MODELS_LOADED = Gauge(
    "milo_models_loaded",
    "Number of loaded ML models",
)

GPU_VRAM_USED = Gauge(
    "milo_gpu_vram_used_bytes",
    "GPU VRAM used in bytes",
)

LLM_REQUESTS = Counter(
    "milo_llm_requests_total",
    "Total LLM requests",
    ["mode"],
)

LLM_FALLBACK = Counter(
    "milo_llm_fallback_total",
    "LLM fallback from cloud to local",
)

ACTIVE_SESSIONS = Gauge(
    "milo_active_sessions",
    "Active conversation sessions",
)

ACTIVE_WEBSOCKETS = Gauge(
    "milo_active_websockets",
    "Active WebSocket connections",
)


def setup_metrics(app):
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/docs", "/openapi.json"],
    ).instrument(app).expose(app, endpoint="/metrics")
