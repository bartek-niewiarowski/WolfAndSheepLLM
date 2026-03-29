from __future__ import annotations

import random
import re
import time
from typing import Optional
import json

import requests
from google import genai
from google.api_core import exceptions as google_exceptions

from game import Move, Player, WolfAndSheepGame


SYSTEM_PROMPT = """Jesteś silnym graczem w grze Wilk i Owce.

Zasady gry:
- Plansza ma współrzędne (wiersz, kolumna)
- Ruchy są po przekątnych (ukośnie)
- OWCE (S):
  - poruszają się tylko do przodu (większy numer wiersza)
- WILK (W):
  - może poruszać się w każdym kierunku ukośnie (do przodu i do tyłu)

Cel gry:
- Wilk wygrywa jeśli dotrze do wiersza 0
- Owce wygrywają jeśli zablokują wilka (brak ruchów)

Twoje zadanie:
1. Przeanalizuj planszę
2. Rozważ wszystkie legalne ruchy
3. Wybierz najlepszy ruch

Zasady odpowiedzi:
- Odpowiedz WYŁĄCZNIE poprawnym JSON-em
- Format odpowiedzi:
{"move_index": LICZBA}

Wymagania:
- LICZBA musi być jednym z podanych indeksów
- nie dodawaj żadnego innego tekstu
- nie używaj markdown
- nie zwracaj wyjaśnień

WAŻNE:
- Wilk może poruszać się zarówno do przodu jak i do tyłu
- upewnij się, że rozważasz wszystkie kierunki ruchu
"""


def parse_move_index(response_text: str, num_moves: int) -> Optional[int]:
    text = response_text.strip()

    # 1. Idealny przypadek: sama liczba
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < num_moves:
            return idx

    # 2. JSON typu {"move_index": 2}
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "move_index" in data:
            idx = int(data["move_index"])
            if 0 <= idx < num_moves:
                return idx
    except Exception:
        pass

    # 3. Szukaj wszystkich liczb i wybierz pierwszą, która mieści się w zakresie ruchów
    matches = re.findall(r"\d+", text)
    for m in matches:
        idx = int(m)
        if 0 <= idx < num_moves:
            return idx

    return None


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
    ):
        self.player = player
        self.backend = backend.lower()
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.request_timeout = request_timeout
        self.ollama_base_url = ollama_base_url.rstrip("/")

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
        else:
            raise ValueError("backend musi mieć wartość 'vertex' albo 'ollama'.")

    def choose_move(self, game: WolfAndSheepGame) -> Optional[Move]:
        legal_moves = game.get_all_valid_moves(self.player)
        if not legal_moves:
            return None

        prompt = self._build_prompt(game)

        if self.verbose:
            print(f"\n[LLMAgent/{self.player.value}] Backend: {self.backend}")
            print(f"[LLMAgent/{self.player.value}] Model: {self.model}")
            print(f"[LLMAgent/{self.player.value}] Prompt:\n{prompt}\n")

        response_text = self._call_llm(prompt)

        if self.verbose:
            print(f"[LLMAgent/{self.player.value}] Odpowiedź LLM: {response_text!r}")

        idx = parse_move_index(response_text, len(legal_moves))

        if idx is None:
            raise ValueError(
                f"[LLMAgent/{self.player.value}] Nie udało się sparsować odpowiedzi LLM.\n"
                f"Odpowiedź: {response_text!r}\n"
                f"Liczba legalnych ruchów: {len(legal_moves)}\n"
                f"Dozwolone indeksy: 0..{len(legal_moves)-1}"
            )

        return legal_moves[idx]

    def _build_prompt(self, game: WolfAndSheepGame) -> str:
        return game.to_prompt_format(self.player)

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "vertex":
            return self._call_vertex(prompt)
        if self.backend == "ollama":
            return self._call_ollama(prompt)
        raise RuntimeError(f"Nieobsługiwany backend: {self.backend}")

    def _call_vertex(self, prompt: str) -> str:
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

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_base_url}/api/chat"

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
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