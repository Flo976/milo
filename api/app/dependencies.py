import logging
from typing import Annotated

import redis.asyncio as redis
from fastapi import Depends, Header, HTTPException, status

from app.config import settings
from app.models.model_manager import model_manager
from app.services.session import SessionManager

logger = logging.getLogger("milo")

_redis_pool: redis.Redis | None = None
_session_manager: SessionManager | None = None


async def get_redis() -> redis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.from_url(
            settings.redis_url, decode_responses=True
        )
    return _redis_pool


async def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        r = await get_redis()
        _session_manager = SessionManager(r)
    return _session_manager


def get_model_manager():
    return model_manager


async def verify_api_key(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme",
        )
    if token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return token


RedisDep = Annotated[redis.Redis, Depends(get_redis)]
SessionDep = Annotated[SessionManager, Depends(get_session_manager)]
AuthDep = Annotated[str, Depends(verify_api_key)]
