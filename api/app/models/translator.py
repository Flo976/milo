import logging
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from app.config import settings
from app.models.model_manager import model_manager

logger = logging.getLogger("milo")

# NLLB language codes
LANG_MAP = {
    "mg": "plt_Latn",  # Malagasy (Plateau)
    "fr": "fra_Latn",  # French
}


class NLLBTranslator:
    def __init__(self, tokenizer, model, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._lock = threading.Lock()  # tokenizer.src_lang is mutable shared state

    def translate(self, text: str, source: str, target: str) -> str:
        """Translate text between mg and fr."""
        src_lang = LANG_MAP.get(source, source)
        tgt_lang = LANG_MAP.get(target, target)

        with self._lock:
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_new_tokens=128,
                )

            result = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
        return result


def load_translator() -> NLLBTranslator:
    logger.info("Loading NLLB translator: %s", settings.nllb_model_id)
    device = model_manager.device

    tokenizer = NllbTokenizer.from_pretrained(settings.nllb_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.nllb_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    translator = NLLBTranslator(tokenizer, model, device)
    model_manager.register("translator", translator)
    return translator
