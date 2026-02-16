import json
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings

# Paths that don't require auth
PUBLIC_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


class AuthMiddleware:
    """Pure ASGI middleware — works with both HTTP and WebSocket."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # Only intercept HTTP requests; let WebSocket and lifespan pass through
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        # Skip auth for public paths
        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()

        if not auth.lower().startswith("bearer "):
            await self._send_json(send, 401, {"detail": "Missing or invalid Authorization header"})
            return

        token = auth[7:]
        if token != settings.api_key:
            await self._send_json(send, 401, {"detail": "Invalid API key"})
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _send_json(send: Send, status: int, body: dict):
        payload = json.dumps(body).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(payload)).encode()],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": payload,
        })
