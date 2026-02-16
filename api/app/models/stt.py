import logging
import math

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

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe 16kHz mono float32 audio to text.

        Returns (text, confidence) where confidence is the mean token
        probability from the generation scores (0.0–1.0).
        """
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
            output = self.model.generate(
                input_features,
                language="mg",
                task="transcribe",
                max_new_tokens=128,
                output_scores=True,
                return_dict_in_generate=True,
            )

        predicted_ids = output.sequences
        scores = output.scores  # tuple of tensors, one per generated token

        logger.info("[STT] Predicted IDs shape: %s, tokens: %s", predicted_ids.shape, predicted_ids[0][:20].tolist())

        text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        # Compute confidence from token probabilities
        confidence = self._compute_confidence(predicted_ids, scores)

        logger.info("[STT] Text: '%s' (confidence=%.3f)", text, confidence)
        return text, confidence

    def _compute_confidence(self, sequences: torch.Tensor, scores: tuple) -> float:
        """Compute mean token probability as a confidence score."""
        if not scores:
            return 0.0

        eos_id = self.processor.tokenizer.eos_token_id
        generated_tokens = sequences[0, -len(scores):]

        token_probs = []
        for i, score in enumerate(scores):
            token_id = generated_tokens[i].item()
            # Skip EOS token — it inflates confidence artificially
            if token_id == eos_id:
                continue
            probs = torch.softmax(score[0].float(), dim=-1)
            token_probs.append(probs[token_id].item())

        if not token_probs:
            return 0.0

        # Geometric mean of probabilities (more robust to outliers)
        avg_log_prob = sum(math.log(max(p, 1e-10)) for p in token_probs) / len(token_probs)
        confidence = math.exp(avg_log_prob)
        return max(0.0, min(1.0, confidence))


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
