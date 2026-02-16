import logging

from app.config import settings

logger = logging.getLogger("milo")


class FallbackManager:
    """Manages cloud/local mode switching for LLM inference."""

    def __init__(self):
        self._mode = settings.default_mode
        self._consecutive_cloud_failures = 0
        self._failure_threshold = 3

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_cloud(self) -> bool:
        return self._mode == "cloud"

    def set_mode(self, mode: str):
        if mode not in ("cloud", "local"):
            raise ValueError(f"Invalid mode: {mode}")
        old = self._mode
        self._mode = mode
        if old != mode:
            logger.info("Mode switched: %s -> %s", old, mode)

    def report_cloud_success(self):
        self._consecutive_cloud_failures = 0

    def report_cloud_failure(self):
        self._consecutive_cloud_failures += 1
        logger.warning(
            "Cloud failure #%d", self._consecutive_cloud_failures
        )
        if self._consecutive_cloud_failures >= self._failure_threshold:
            logger.warning(
                "Too many cloud failures, switching to local mode"
            )
            self._mode = "local"

    def should_try_cloud(self) -> bool:
        if self._mode == "local":
            return False
        # Skip cloud if API key is missing or placeholder
        key = settings.anthropic_api_key
        if not key or key.startswith("sk-ant-xxxxx") or len(key) < 20:
            return False
        return True

    def status(self) -> dict:
        return {
            "mode": self._mode,
            "consecutive_cloud_failures": self._consecutive_cloud_failures,
            "cloud_available": self.should_try_cloud(),
        }


fallback_manager = FallbackManager()
