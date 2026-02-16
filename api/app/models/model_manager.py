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

    def register(self, name: str, model: Any):
        self._models[name] = model
        logger.info("Model registered: %s", name)

    def unload(self, name: str):
        model = self._models.pop(name, None)
        if model is not None:
            del model
            if self._device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded: %s", name)

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
