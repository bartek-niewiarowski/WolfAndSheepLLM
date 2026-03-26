from __future__ import annotations

import re
import time
from typing import Optional
from google import genai
from game import Player, WolfAndSheepGame, Move

SYSTEM_PROMPT = """Jesteś ekspertem w grze Wilk i Owce na planszy szachowej.

Zasady gry:
- Plansza jest szachowa, figury poruszają się tylko po ciemnych polach (ukośnie)
- OWCE (S) zaczynają w pierwszym rzędzie (rząd 0) i mogą poruszać się TYLKO do przodu (rosnące numery rzędów)
- WILK (W) zaczyna w ostatnim rzędzie i może poruszać się w KAŻDYM kierunku ukośnie
- Wilk wygrywa jeśli dotrze do rzędu 0
- Owce wygrywają jeśli zablokują wilka tak, że nie ma żadnego ruchu

Twoje zadanie:
- Otrzymasz stan planszy i listę legalnych ruchów z indeksami
- Wybierz najlepszy ruch i odpowiedz WYŁĄCZNIE numerem indeksu ruchu (np. "2")
- Nie dodawaj żadnych wyjaśnień ani dodatkowego tekstu - tylko sam numer
"""

def parse_move_index(response_text: str, num_moves: int) -> Optional[int]:
    """Wyciąga indeks ruchu z odpowiedzi LLM."""
    text = response_text.strip()
    
    # Szukaj pierwszej liczby w odpowiedzi
    match = re.search(r'\d+', text)
    if match:
        idx = int(match.group())
        if 0 <= idx < num_moves:
            return idx
    
    return None


class LLMAgent:
    def __init__(
        self,
        player: Player,
        project: str,
        location: str = "global",
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.2,
        verbose: bool = False,
    ):
        self.player = player
        self.model = model
        self.temperature = temperature
        self.verbose = verbose

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    def choose_move(self, game: WolfAndSheepGame) -> Optional[Move]:
        legal_moves = game.get_all_valid_moves(self.player)
        if not legal_moves:
            return None

        prompt = self._build_prompt(game)

        if self.verbose:
            print(f"\n[LLMAgent/{self.player.value}] Prompt:\n{prompt}\n")

        response_text = self._call_llm(prompt)

        if self.verbose:
            print(f"[LLMAgent/{self.player.value}] Odpowiedź LLM: {response_text!r}")

        idx = parse_move_index(response_text, len(legal_moves))

        if idx is None:
            if self.verbose:
                print(f"[LLMAgent/{self.player.value}] Nie udało się sparsować odpowiedzi, wybieram ruch 0")
            return legal_moves[0]

        return legal_moves[idx]

    def _build_prompt(self, game: WolfAndSheepGame) -> str:
        return game.to_prompt_format(self.player)

    def _call_llm(self, prompt: str) -> str:
        time.sleep(2)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=16,
            ),
        )
        return response.text.strip()