import logging

import numpy as np
import torch

from app.models.model_manager import model_manager

logger = logging.getLogger("milo")


class SileroVAD:
    SAMPLE_RATE = 16000

    def __init__(self, model, utils):
        self.model = model
        self._get_speech_timestamps = utils[0]
        self._threshold = 0.5

    def detect_speech(self, audio: np.ndarray) -> list[dict]:
        """Detect speech segments in audio. Returns list of {start, end} in samples."""
        tensor = torch.from_numpy(audio).float()
        timestamps = self._get_speech_timestamps(
            tensor,
            self.model,
            sampling_rate=self.SAMPLE_RATE,
            threshold=self._threshold,
        )
        return timestamps

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if a chunk contains speech by splitting into 512-sample windows."""
        if len(audio_chunk) < 512:
            return False
        # Silero VAD requires exactly 512 samples at 16kHz
        max_conf = 0.0
        for start in range(0, len(audio_chunk) - 511, 512):
            window = torch.from_numpy(audio_chunk[start:start + 512].copy()).float()
            conf = self.model(window, self.SAMPLE_RATE).item()
            if conf > max_conf:
                max_conf = conf
        return max_conf > self._threshold


def load_vad() -> SileroVAD:
    logger.info("Loading Silero VAD v5")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    vad = SileroVAD(model, utils)
    model_manager.register("vad", vad)
    return vad
