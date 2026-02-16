import asyncio
import logging
from typing import Any

import torch

logger = logging.getLogger("milo")


class ModelManager:
    """Lazy-loads ML models on first access with a GPU semaphore."""

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._gpu_semaphore = asyncio.Semaphore(2)
        self._load_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def gpu_semaphore(self) -> asyncio.Semaphore:
        return self._gpu_semaphore

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def get(self, name: str) -> Any | None:
        return self._models.get(name)

    async def get_or_load(self, name: str, loader) -> Any:
        """Get a model, loading it if needed with a per-model lock to prevent races."""
        model = self._models.get(name)
        if model is not None:
            return model

        # Get or create a per-model lock
        async with self._global_lock:
            if name not in self._load_locks:
                self._load_locks[name] = asyncio.Lock()
            lock = self._load_locks[name]

        async with lock:
            # Double-check after acquiring lock
            model = self._models.get(name)
            if model is not None:
                return model
            model = await asyncio.to_thread(loader)
            return model

    def register(self, name: str, model: Any):
        self._models[name] = model
        logger.info("Model registered: %s", name)
        self._update_metrics()

    def unload(self, name: str):
        model = self._models.pop(name, None)
        if model is not None:
            del model
            if self._device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded: %s", name)
            self._update_metrics()

    def _update_metrics(self):
        """Update Prometheus gauges for models loaded and GPU VRAM."""
        try:
            from app.middleware.metrics import MODELS_LOADED, GPU_VRAM_USED
            MODELS_LOADED.set(len(self._models))
            if self._device == "cuda":
                GPU_VRAM_USED.set(torch.cuda.memory_allocated())
        except Exception as e:
            logger.debug("Metrics update skipped: %s", e)

    def unload_all(self):
        for name in list(self._models.keys()):
            self.unload(name)

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def vram_usage(self) -> dict:
        if self._device != "cuda":
            return {"available": False}
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
        return {
            "available": True,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2),
        }


model_manager = ModelManager()
