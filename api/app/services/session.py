import json
import logging
import uuid

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger("milo")


class SessionManager:
    PREFIX = "milo:session:"

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"{self.PREFIX}{session_id}"

    async def create(self) -> str:
        session_id = str(uuid.uuid4())
        await self._redis.set(
            self._key(session_id),
            json.dumps({"history": []}),
            ex=settings.session_ttl_s,
        )
        return session_id

    async def get_history(self, session_id: str) -> list[dict]:
        data = await self._redis.get(self._key(session_id))
        if data is None:
            return []
        return json.loads(data).get("history", [])

    async def add_exchange(
        self, session_id: str, user_msg: str, assistant_msg: str
    ):
        key = self._key(session_id)
        data = await self._redis.get(key)
        if data is None:
            history = []
        else:
            history = json.loads(data).get("history", [])

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

        # Keep only last N exchanges
        max_items = settings.session_max_history * 2
        if len(history) > max_items:
            history = history[-max_items:]

        await self._redis.set(
            key,
            json.dumps({"history": history}),
            ex=settings.session_ttl_s,
        )

    async def exists(self, session_id: str) -> bool:
        return await self._redis.exists(self._key(session_id)) > 0

    async def delete(self, session_id: str):
        await self._redis.delete(self._key(session_id))
