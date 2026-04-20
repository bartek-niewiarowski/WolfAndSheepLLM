from __future__ import annotations

import random
import re
import time
from typing import Optional
import json
from dataclasses import asdict, dataclass

import requests
from google import genai
from google.api_core import exceptions as google_exceptions

import os
from openai import OpenAI, RateLimitError, APITimeoutError, APIError, APIConnectionError

from game import Move, Player, WolfAndSheepGame

class LLMCallError(Exception):
    def __init__(self, message: str, error_type: str = "llm_error"):
        super().__init__(message)
        self.error_type = error_type


SYSTEM_PROMPT = """You are playing Wolf and Sheep. Your job is to avoid bad moves.

Rules:
- Play only on dark squares and move diagonally.
- Sheep (S): move forward only (row increases).
- Wolf (W): move diagonally in any direction.
- Wolf wins by reaching row 0.
- Sheep win by blocking the wolf.

Decision policy:
- First, avoid moves that immediately worsen your position.
- Avoid moves that reduce your future options unless necessary.
- Avoid moves that allow the opponent an obvious advantage on the next turn.
- Then choose the best remaining move.

Input:
You will get the current board and a list of legal moves.

Output:
Return ONLY the index of one legal move.
Example: 0
No explanation.
"""

@dataclass
class PromptPayload:
    system_prompt: str
    user_prompt: str

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        return (
            f"[SYSTEM]\n{self.system_prompt}\n\n"
            f"[USER]\n{self.user_prompt}"
        )

def parse_move_index(response_text: str, num_moves: int) -> Optional[int]:
    text = response_text.strip()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < num_moves:
            return idx
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "move_index" in data:
            idx = int(data["move_index"])
            if 0 <= idx < num_moves:
                return idx
    except Exception:
        pass

    matches = re.findall(r"\d+", text)
    for m in matches:
        idx = int(m)
        if 0 <= idx < num_moves:
            return idx

    return None

def extract_text_from_response(response) -> str:
    text = (getattr(response, "output_text", None) or "").strip()
    if text:
        return text

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    t = getattr(content, "text", None)
                    if t:
                        return t.strip()
    return ""


class LLMAgent:
    def __init__(
        self,
        player: Player,
        backend: str = "vertex",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        verbose: bool = False,
        project: Optional[str] = None,
        location: str = "global",
        ollama_base_url: str = "http://localhost:11434",
        request_timeout: int = 120,
        openai_api_key: Optional[str] = None,
    ):
        self.player = player
        self.backend = backend.lower()
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.request_timeout = request_timeout
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.openai_client: Optional[OpenAI] = None
        self.last_prompt: Optional[PromptPayload] = None
        self.prompt_history: list[PromptPayload] = []

        self.client: Optional[genai.Client] = None
        if self.backend == "vertex":
            if not project:
                raise ValueError("Dla backend='vertex' musisz podać project.")
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        elif self.backend == "ollama":
            pass
        elif self.backend == "openai":
            self.openai_client = OpenAI(
                api_key = os.environ.get("OPENAI_API_KEY"),
                timeout=request_timeout,
            )
        else:
            raise ValueError("backend musi mieć wartość 'vertex', 'openai' albo 'ollama'.")

    def choose_move(self, game: WolfAndSheepGame) -> Optional[Move]:
        legal_moves = game.get_all_valid_moves(self.player)
        if not legal_moves:
            return None

        prompt = self._build_prompt(game)
        self.last_prompt = prompt
        self.prompt_history.append(prompt)

        if self.verbose:
            print(f"\n[LLMAgent/{self.player.value}] Backend: {self.backend}")
            print(f"[LLMAgent/{self.player.value}] Model: {self.model}")
            print(f"[LLMAgent/{self.player.value}] Prompt:\n{prompt.pretty()}\n")

        try:
            response_text = self._call_llm(prompt)
        except Exception as e:
            raise LLMCallError(
                f"[LLMAgent/{self.player.value}] Błąd wywołania LLM: {e}",
                error_type="llm_call_failed"
            ) from e

        if self.verbose:
            print(f"[LLMAgent/{self.player.value}] Odpowiedź LLM: {response_text!r}")

        idx = parse_move_index(response_text, len(legal_moves))

        if idx is None:
            raise LLMCallError(
            f"[LLMAgent/{self.player.value}] Nie udało się sparsować odpowiedzi LLM. "
            f"Odpowiedź: {response_text!r}, liczba legalnych ruchów: {len(legal_moves)}",
            error_type="invalid_llm_output"
            )

        return legal_moves[idx]

    def _build_prompt(self, game: WolfAndSheepGame) -> PromptPayload:
        return PromptPayload(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=game.to_prompt_format(self.player),
        )

    def _call_llm(self, prompt: PromptPayload) -> str:
        if self.backend == "vertex":
            return self._call_vertex(prompt)
        if self.backend == "ollama":
            return self._call_ollama(prompt)
        if self.backend == "openai":
            return self._call_openai(prompt)
        raise RuntimeError(f"Nieobsługiwany backend: {self.backend}")

    def _call_vertex(self, prompt: PromptPayload) -> str:
        if self.client is None:
            raise RuntimeError("Vertex client nie został zainicjalizowany.")

        base_delay = 2.0
        max_delay = 30.0
        last_exc = None

        for attempt in range(6):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_output_tokens=16,
                    ),
                )
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Pusta odpowiedź z modelu")
                return text.strip()

            except google_exceptions.ResourceExhausted as e:
                last_exc = e
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.25)
                sleep_for = delay + jitter
                print(f"[Vertex] 429 RESOURCE_EXHAUSTED, retry {attempt + 1}/6 za {sleep_for:.1f}s")
                time.sleep(sleep_for)

        raise RuntimeError(f"Vertex LLM failed after retries: {last_exc}")

    def _call_ollama(self, prompt: PromptPayload) -> str:
        url = f"{self.ollama_base_url}/api/chat"

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": 8,
            },
        }

        base_delay = 1.0
        max_delay = 10.0
        last_exc = None

        for attempt in range(4):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()

                text = data.get("message", {}).get("content", "").strip()
                if not text:
                    raise RuntimeError("Pusta odpowiedź z Ollama")

                return text

            except (requests.RequestException, RuntimeError) as e:
                last_exc = e
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.25)
                sleep_for = delay + jitter
                print(f"[Ollama] błąd wywołania, retry {attempt + 1}/4 za {sleep_for:.1f}s")
                time.sleep(sleep_for)

        raise RuntimeError(f"Ollama LLM failed after retries: {last_exc}")
    
    def _call_openai(self, prompt: PromptPayload) -> str:
        if self.openai_client is None:
            raise RuntimeError("OpenAI client nie został zainicjalizowany.")

        try:
            response = self.openai_client.responses.create(
                model=self.model,
                instructions=prompt.system_prompt,
                input=[{"role": "user", "content": prompt.user_prompt}],
                reasoning={"effort": "low"},
                max_output_tokens=2048,
                text={
                    "format": {
                        "type": "json_schema",
                        "strict": True,
                        "name": "move_choice",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "move_index": {
                                    "type": "integer"
                                }
                            },
                            "required": ["move_index"],
                            "additionalProperties": False
                        }
                    }
                }
            )
        except RateLimitError as e:
            raise RuntimeError(f"OpenAI RateLimitError: {e}") from e
        except APITimeoutError as e:
            raise RuntimeError(f"OpenAI APITimeoutError: {e}") from e
        except APIConnectionError as e:
            raise RuntimeError(f"OpenAI APIConnectionError: {e}") from e
        except APIError as e:
            raise RuntimeError(f"OpenAI APIError: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Nieznany błąd OpenAI: {e}") from e

        if response.status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            raise RuntimeError(f"Odpowiedź incomplete: {reason}")

        raw = extract_text_from_response(response)
        if not raw:
            raise RuntimeError(f"Brak tekstu. status={response.status}, output={response.output}")

        return raw