import logging
import os

from app.config import settings
from app.models.model_manager import model_manager

logger = logging.getLogger("milo")

SYSTEM_PROMPT = (
    "Ianao dia Milo, mpanampy amin'ny feo amin'ny teny malagasy. "
    "Valio amin'ny teny malagasy foana ny fanontaniana rehetra. "
    "Aza mamaly amin'ny teny frantsay na anglisy. "
    "Ataovy fohy ny valinao (fehezanteny 1-3)."
)


class LocalLLM:
    def __init__(self, llm):
        self._llm = llm

    def generate(self, message: str, history: list[dict]) -> str:
        """Generate response using local LLM via llama-cpp-python."""
        import time
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})

        start = time.perf_counter()
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=60,
            temperature=0.5,
            top_p=0.9,
            stop=["\n\n", "* ", "- ", "\n*", "\n-"],
        )
        elapsed = (time.perf_counter() - start) * 1000

        reply = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        logger.info("[LLM-LOCAL] %dms | prompt=%d completion=%d | '%s'",
                    round(elapsed),
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    reply[:100])
        return reply


def load_llm_local() -> LocalLLM:
    model_path = settings.local_llm_model_path
    logger.info("Loading local LLM: %s", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Local LLM model not found: {model_path}. "
            "Download it first."
        )

    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=settings.local_llm_n_gpu_layers,
        n_ctx=4096,
        n_batch=512,
        flash_attn=True,
        verbose=False,
    )

    local = LocalLLM(llm)
    model_manager.register("llm_local", local)
    return local
