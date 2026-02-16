import time

import redis.asyncio as redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings

# Paths exempt from rate limiting
EXEMPT_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str = ""):
        super().__init__(app)
        self._redis: redis.Redis | None = None
        self._redis_url = redis_url or settings.redis_url

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._redis

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in EXEMPT_PATHS:
            return await call_next(request)

        # Extract API key for tier identification
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not token:
            return await call_next(request)

        # Determine rate limit based on key (simplified: single tier)
        limit = settings.rate_limit_free

        try:
            r = await self._get_redis()
            key = f"milo:rate:{token}"
            window = 3600  # 1 hour

            current = await r.get(key)
            if current is not None and int(current) >= limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded: {limit} requests/hour"
                    },
                    headers={"Retry-After": str(window)},
                )

            pipe = r.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            await pipe.execute()

        except Exception:
            # If Redis is down, allow the request
            pass

        return await call_next(request)
