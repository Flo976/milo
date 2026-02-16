import io
import logging

import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer

from app.config import settings
from app.models.model_manager import model_manager

logger = logging.getLogger("milo")


class MMSTTS:
    def __init__(self, tokenizer, model, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV bytes."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        waveform = output.waveform[0].cpu().float().numpy()
        sr = self.model.config.sampling_rate

        buf = io.BytesIO()
        sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()


def load_tts() -> MMSTTS:
    logger.info("Loading MMS-TTS: %s", settings.mms_tts_model_id)
    device = model_manager.device

    tokenizer = AutoTokenizer.from_pretrained(settings.mms_tts_model_id)
    model = VitsModel.from_pretrained(settings.mms_tts_model_id).to(device)
    model.eval()

    tts = MMSTTS(tokenizer, model, device)
    model_manager.register("tts", tts)
    return tts
