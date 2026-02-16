"""Test and interact with Mistral 7B via the llama.cpp OpenAI-compatible API.

This script provides:
  - A health check for the llama-server
  - A simple chat completion test
  - A streaming chat completion test
  - A reusable MistralClient class for integration in the Milo pipeline

Usage:
  # Basic test (server must be running on port 8080):
  python3 /home/florent/milo/scripts/05b_test_mistral.py

  # Custom server URL:
  python3 /home/florent/milo/scripts/05b_test_mistral.py --url http://localhost:8080

  # Interactive chat mode:
  python3 /home/florent/milo/scripts/05b_test_mistral.py --interactive

  # Single prompt:
  python3 /home/florent/milo/scripts/05b_test_mistral.py --prompt "Explain Madagascar in 3 sentences."
"""

import argparse
import json
import sys
import time
from typing import Generator, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


# =============================================================================
# MistralClient -- Reusable wrapper for the llama.cpp OpenAI-compatible API
# =============================================================================

class MistralClient:
    """Client for llama.cpp server with OpenAI-compatible API.

    This class uses only the standard library (urllib) so it has zero
    external dependencies. For production use, consider switching to
    the `openai` Python package or `httpx`.

    Usage:
        client = MistralClient("http://localhost:8080")
        response = client.chat("Hello, how are you?")
        print(response)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "mistral-7b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # -- Health check ---------------------------------------------------------

    def is_alive(self) -> bool:
        """Check if the llama-server is running and responsive."""
        try:
            req = Request(f"{self.base_url}/health")
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data.get("status") == "ok"
        except (URLError, ConnectionError, OSError, json.JSONDecodeError):
            return False

    def wait_for_server(self, max_wait: int = 60, interval: int = 2) -> bool:
        """Wait until the server is ready, up to max_wait seconds."""
        start = time.time()
        while time.time() - start < max_wait:
            if self.is_alive():
                return True
            time.sleep(interval)
        return False

    # -- Chat completion (non-streaming) --------------------------------------

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Send a chat completion request and return the assistant's reply.

        Args:
            user_message: The user's message.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode())

        return result["choices"][0]["message"]["content"]

    # -- Chat completion (streaming) ------------------------------------------

    def chat_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Generator[str, None, None]:
        """Send a streaming chat completion request, yielding tokens as they arrive.

        Args:
            user_message: The user's message.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Token strings as they are generated.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(req, timeout=self.timeout) as resp:
            buffer = ""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                # Process complete SSE lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload_str = line[6:]  # Remove "data: " prefix
                    if payload_str == "[DONE]":
                        return
                    try:
                        event = json.loads(payload_str)
                        delta = event["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # -- Multi-turn conversation ----------------------------------------------

    def chat_multi(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Send a multi-turn conversation and return the assistant's reply.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode())

        return result["choices"][0]["message"]["content"]

    # -- Raw completion (non-chat) --------------------------------------------

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        """Send a raw text completion request (non-chat endpoint).

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated completion text.
        """
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}/completion",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode())

        return result.get("content", "")


# =============================================================================
# Test functions
# =============================================================================

def test_health(client: MistralClient) -> bool:
    """Test that the server is alive."""
    print("=" * 60)
    print("TEST 1: Health check")
    print("=" * 60)

    alive = client.is_alive()
    if alive:
        print("  [PASS] Server is alive at {}".format(client.base_url))
    else:
        print("  [FAIL] Server not responding at {}".format(client.base_url))
        print("")
        print("  Make sure llama-server is running:")
        print("    /home/florent/milo/llama.cpp/build/bin/llama-server \\")
        print("      -m /home/florent/milo/models/mistral-7b/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \\")
        print("      --host 0.0.0.0 --port 8080 -ngl 99 -c 8192")
    print("")
    return alive


def test_chat(client: MistralClient) -> bool:
    """Test a simple chat completion."""
    print("=" * 60)
    print("TEST 2: Chat completion")
    print("=" * 60)

    prompt = "In one sentence, what is the capital of Madagascar?"
    print("  Prompt: {}".format(prompt))
    print("")

    try:
        start = time.time()
        response = client.chat(prompt, max_tokens=128)
        elapsed = time.time() - start

        print("  Response: {}".format(response.strip()))
        print("  Time: {:.1f}s".format(elapsed))
        print("  [PASS]")
        print("")
        return True
    except Exception as e:
        print("  [FAIL] {}".format(e))
        print("")
        return False


def test_streaming(client: MistralClient) -> bool:
    """Test streaming chat completion."""
    print("=" * 60)
    print("TEST 3: Streaming chat completion")
    print("=" * 60)

    prompt = "Count from 1 to 5 in Malagasy."
    print("  Prompt: {}".format(prompt))
    print("  Response: ", end="", flush=True)

    try:
        start = time.time()
        token_count = 0
        for token in client.chat_stream(prompt, max_tokens=128):
            print(token, end="", flush=True)
            token_count += 1
        elapsed = time.time() - start

        print("")
        print("  Tokens: {}, Time: {:.1f}s, Speed: {:.1f} tok/s".format(
            token_count, elapsed, token_count / elapsed if elapsed > 0 else 0
        ))
        print("  [PASS]")
        print("")
        return True
    except Exception as e:
        print("")
        print("  [FAIL] {}".format(e))
        print("")
        return False


def test_system_prompt(client: MistralClient) -> bool:
    """Test chat with a system prompt (pipeline integration test)."""
    print("=" * 60)
    print("TEST 4: System prompt (pipeline integration)")
    print("=" * 60)

    system = (
        "You are Milo, a helpful voice assistant specialized in the Malagasy language. "
        "You respond concisely and helpfully. If the user speaks in Malagasy, respond in Malagasy. "
        "Otherwise respond in the language they use."
    )
    user_msg = "Bonjour Milo, comment dit-on 'merci' en malagasy ?"

    print("  System: {}...".format(system[:80]))
    print("  User: {}".format(user_msg))
    print("")

    try:
        start = time.time()
        response = client.chat(user_msg, system_prompt=system, max_tokens=256)
        elapsed = time.time() - start

        print("  Milo: {}".format(response.strip()))
        print("  Time: {:.1f}s".format(elapsed))
        print("  [PASS]")
        print("")
        return True
    except Exception as e:
        print("  [FAIL] {}".format(e))
        print("")
        return False


def interactive_mode(client: MistralClient):
    """Interactive chat loop for manual testing."""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("  Type your messages. Type 'quit' or Ctrl+C to exit.")
    print("  Type '/system <prompt>' to set a system prompt.")
    print("  Type '/clear' to reset the conversation.")
    print("")

    system_prompt = (
        "You are Milo, a helpful voice assistant specialized in the Malagasy language. "
        "You respond concisely and helpfully."
    )
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.startswith("/system "):
            system_prompt = user_input[8:]
            print("  [System prompt updated]")
            continue
        if user_input == "/clear":
            history = []
            print("  [Conversation cleared]")
            continue

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        print("Milo: ", end="", flush=True)
        try:
            response = client.chat_multi(messages, max_tokens=512)
            print(response.strip())

            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            # Keep history manageable (last 10 turns)
            if len(history) > 20:
                history = history[-20:]
        except Exception as e:
            print("[Error: {}]".format(e))

        print("")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Mistral 7B via llama.cpp OpenAI-compatible API"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="llama-server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt to send (non-interactive)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt to use",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait N seconds for server to be ready before testing",
    )
    args = parser.parse_args()

    client = MistralClient(base_url=args.url)

    # Wait for server if requested
    if args.wait > 0:
        print("Waiting up to {}s for server at {}...".format(args.wait, args.url))
        if not client.wait_for_server(max_wait=args.wait):
            print("Server did not become ready. Exiting.")
            sys.exit(1)
        print("Server is ready!")
        print("")

    # Single prompt mode
    if args.prompt:
        if not client.is_alive():
            print("Error: Server not responding at {}".format(args.url))
            sys.exit(1)
        response = client.chat(args.prompt, system_prompt=args.system)
        print(response.strip())
        sys.exit(0)

    # Interactive mode
    if args.interactive:
        if not client.is_alive():
            print("Error: Server not responding at {}".format(args.url))
            sys.exit(1)
        interactive_mode(client)
        sys.exit(0)

    # Default: run all tests
    print("")
    print("============================================================")
    print("  MILO -- Mistral 7B API Test Suite")
    print("  Server: {}".format(args.url))
    print("============================================================")
    print("")

    results = []
    results.append(("Health check", test_health(client)))

    if results[0][1]:  # Only run further tests if server is alive
        results.append(("Chat completion", test_chat(client)))
        results.append(("Streaming", test_streaming(client)))
        results.append(("System prompt", test_system_prompt(client)))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print("  [{}] {}".format(status, name))
    print("")
    print("  {}/{} tests passed.".format(passed, total))
    print("")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
