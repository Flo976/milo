import type { WSEvent } from "./types";

type WSHandler = (event: WSEvent) => void;

export class WSManager {
  private ws: WebSocket | null = null;
  private handlers: WSHandler[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private sessionId = "";
  private apiKey = "";
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  onEvent(handler: WSHandler) {
    this.handlers.push(handler);
    return () => {
      this.handlers = this.handlers.filter((h) => h !== handler);
    };
  }

  connect(apiKey: string, sessionId?: string) {
    // Clean up any pending reconnect
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    // Close existing connection — detach callbacks first to avoid triggering reconnect
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      this.ws.onopen = null;
      this.ws.close();
      this.ws = null;
    }

    this.apiKey = apiKey;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    if (sessionId) this.sessionId = sessionId;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = import.meta.env.VITE_WS_HOST ?? window.location.host;
    const params = new URLSearchParams({ api_key: apiKey });
    if (this.sessionId) params.set("session_id", this.sessionId);

    const url = `${protocol}//${host}/api/v1/conversation?${params}`;
    console.log("[WS] Connecting to", url);

    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      console.log("[WS] Connected");
      this.reconnectAttempts = 0;
    };

    ws.onmessage = (evt) => {
      if (typeof evt.data === "string") {
        const event: WSEvent = JSON.parse(evt.data);
        if (event.type === "session" && event.session_id) {
          this.sessionId = event.session_id;
        }
        this.handlers.forEach((h) => h(event));
      }
    };

    ws.onclose = () => {
      console.log("[WS] Closed");
      // Only reconnect if this is still the active connection
      if (this.ws === ws) {
        this.ws = null;
        this.tryReconnect();
      }
    };

    ws.onerror = (e) => {
      console.error("[WS] Error", e);
    };
  }

  private tryReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts || !this.apiKey) {
      console.log("[WS] Not reconnecting (attempts=%d, apiKey=%s)", this.reconnectAttempts, !!this.apiKey);
      return;
    }
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log("[WS] Reconnecting in %dms (attempt %d/%d)", delay, this.reconnectAttempts, this.maxReconnectAttempts);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect(this.apiKey, this.sessionId);
    }, delay);
  }

  sendAudio(pcmData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(pcmData);
    }
  }

  sendControl(msg: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.apiKey = "";
    this.maxReconnectAttempts = 0;
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      this.ws.onopen = null;
      this.ws.close();
      this.ws = null;
    }
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get currentSessionId(): string {
    return this.sessionId;
  }
}

export const wsManager = new WSManager();
