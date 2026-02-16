import asyncio
import logging

import anthropic

from app.config import settings
from app.models.model_manager import model_manager

logger = logging.getLogger("milo")

SYSTEM_PROMPT = (
    "Ianao dia Milo, mpanampy amin'ny feo amin'ny teny malagasy. "
    "Valio amin'ny teny malagasy foana ny fanontaniana rehetra. "
    "Aza mamaly amin'ny teny frantsay na anglisy raha tsy nangatahina manokana. "
    "Tsarovy ny zavatra rehetra nolazain'ny mpampiasa (anarana, toerana, zavatra tiany...) "
    "ary ampiasao izany rehefa misy ifandraisany amin'ny resaka. "
    "Ataovy fohy ny valinao (fehezanteny 1-3). "
    "Aza mampiasa emoji na sary. "
    "Aza mampiasa teny anglisy na frantsay ao anatin'ny fehezanteny (ohatra: ampiasao 'degre' fa tsy 'degrees')."
)


class ClaudeLLM:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def generate(self, message: str, history: list[dict]) -> str:
        """Generate a response using Claude Haiku with timeout."""
        messages = []
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})

        response = await asyncio.wait_for(
            self.client.messages.create(
                model=settings.claude_model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=messages,
            ),
            timeout=settings.claude_timeout_s,
        )

        return response.content[0].text


def load_llm_cloud() -> ClaudeLLM:
    logger.info("Initializing Claude LLM client")
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    llm = ClaudeLLM(client)
    model_manager.register("llm_cloud", llm)
    return llm
