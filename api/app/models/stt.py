import logging

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from app.config import settings
from app.models.model_manager import model_manager

logger = logging.getLogger("milo")


class WhisperSTT:
    def __init__(self, processor, model, device: str):
        self.processor = processor
        self.model = model
        self.device = device

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16kHz mono float32 audio to text."""
        logger.info("[STT] Input audio: shape=%s, dtype=%s, rms=%.4f",
                     audio.shape, audio.dtype,
                     float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0)

        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(
            device=self.device, dtype=self.model.dtype
        )
        logger.info("[STT] Features: shape=%s, dtype=%s", input_features.shape, input_features.dtype)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language="mg",
                task="transcribe",
                max_new_tokens=128,
            )

        logger.info("[STT] Predicted IDs shape: %s, tokens: %s", predicted_ids.shape, predicted_ids[0][:20].tolist())

        text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        # Also decode without skipping special tokens for debug
        raw_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
        logger.info("[STT] Raw decode: '%s'", raw_text[:200])
        logger.info("[STT] Clean text: '%s'", text)
        return text


def load_stt() -> WhisperSTT:
    logger.info("Loading Whisper STT: %s", settings.whisper_model_id)
    device = model_manager.device

    # Processor (tokenizer + feature extractor) from base model
    # The fine-tuned checkpoint doesn't save the tokenizer vocab
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    logger.info("Processor loaded from openai/whisper-medium (vocab_size=%d)", processor.tokenizer.vocab_size)

    # Model weights from fine-tuned checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(
        settings.whisper_model_id,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    stt = WhisperSTT(processor, model, device)
    model_manager.register("stt", stt)
    return stt
