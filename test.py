from typing import Optional, Any
import json
from datetime import datetime
from pathlib import Path

from game import WolfAndSheepGame, Player
from minimaxAgent import MinimaxAgent
from llmAgent import LLMAgent, LLMCallError

def get_example_prompt_from_agent(agent: LLMAgent) -> Optional[dict[str, Any]]:
    """
    Zwraca przykładowy prompt z agenta:
    1. pierwszy z historii (najlepszy do raportu)
    2. jeśli brak historii -> last_prompt
    """

    # 1. najlepsza opcja: pierwszy prompt z historii
    if hasattr(agent, "prompt_history") and agent.prompt_history:
        prompt = agent.prompt_history[0]
        return prompt.to_dict()

    # 2. fallback: ostatni prompt
    if hasattr(agent, "last_prompt") and agent.last_prompt is not None:
        return agent.last_prompt.to_dict()

    return None

def play_game(
    wolf_agent,
    sheep_agent,
    board_size: int = 8,
    max_turns: int = 200,
    verbose: bool = False,
) -> tuple[Optional[Player], WolfAndSheepGame]:

    game = WolfAndSheepGame(board_size=board_size)
    moves_played = 0

    if verbose:
        print("=== POCZĄTKOWA PLANSZA ===")
        print(game)
        print()

    for _ in range(max_turns):
        if game.is_game_over():
            break

        if game.current_player == Player.WOLF:
            move = wolf_agent.choose_move(game)
        else:
            move = sheep_agent.choose_move(game)

        if move is None:
            break

        game.make_move(move)
        moves_played += 1

    if verbose:
        print("=== KOŃCOWA PLANSZA ===")
        print(game)
        print()

    return game.get_winner(), game, moves_played

def append_jsonl(filepath: str | Path, record: dict[str, Any]) -> None:
    """
    Dopisuje jeden rekord JSON jako osobną linię do pliku .jsonl
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def run_series(
    num_games: int,
    wolf_agent,
    sheep_agent,
    output_file: str = "game_results.jsonl",
    board_size: int = 8,
    max_turns: int = 200,
    verbose_each_game: bool = False,
) -> dict[str, Any]:
    llm_agent: Optional[LLMAgent] = None
    minimax_agent: Optional[MinimaxAgent] = None

    if isinstance(wolf_agent, LLMAgent):
        llm_agent = wolf_agent
    elif isinstance(sheep_agent, LLMAgent):
        llm_agent = sheep_agent

    if isinstance(wolf_agent, MinimaxAgent):
        minimax_agent = wolf_agent
    elif isinstance(sheep_agent, MinimaxAgent):
        minimax_agent = sheep_agent

    if llm_agent is None:
        raise ValueError("W tej serii musi brać udział LLMAgent.")
    if minimax_agent is None:
        raise ValueError("W tej serii musi brać udział MinimaxAgent.")

    llm_wins = 0
    ai_wins = 0
    draws_or_unfinished = 0
    skipped_games = 0
    game_lengths: list[int] = []
    per_game_results: list[dict[str, Any]] = []
    try:
        for game_idx in range(1, num_games + 1):
            winner, game, moves_played = play_game(
                wolf_agent=wolf_agent,
                sheep_agent=sheep_agent,
                board_size=board_size,
                max_turns=max_turns,
                verbose=verbose_each_game,
            )

            game_lengths.append(moves_played)

            if winner is None:
                winner_label = None
                draws_or_unfinished += 1
            else:
                winner_label = winner.value

                llm_player = llm_agent.player
                minimax_player = minimax_agent.player

                if winner == llm_player:
                    llm_wins += 1
                elif winner == minimax_player:
                    ai_wins += 1
            per_game_results.append(
            {
                "game_index": game_idx,
                "status": "completed",
                "winner": winner_label,
                "moves_played": moves_played,
                "error_type": None,
                "error_message": None,
            }
            )

            print(
                f"[{game_idx}/{num_games}] status=completed, "
                f"winner={winner_label}, moves={moves_played}"
            )
    except LLMCallError as e:
        skipped_games += 1

        per_game_results.append(
            {
                "game_index": game_idx,
                "status": "skipped",
                "winner": None,
                "moves_played": None,
                "error_type": getattr(e, "error_type", "llm_error"),
                "error_message": str(e),
            }
        )

        print(
            f"[{game_idx}/{num_games}] status=skipped, "
            f"reason={getattr(e, 'error_type', 'llm_error')} | {e}"
        )

    except Exception as e:
        skipped_games += 1

        per_game_results.append(
            {
                "game_index": game_idx,
                "status": "skipped",
                "winner": None,
                "moves_played": None,
                "error_type": "unexpected_error",
                "error_message": str(e),
            }
        )

        print(
            f"[{game_idx}/{num_games}] status=skipped, "
            f"reason=unexpected_error | {e}"
        )

    example_prompt = get_example_prompt_from_agent(llm_agent)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_games": num_games,
        "board_size": board_size,
        "max_turns": max_turns,
        "llm": {
            "player": llm_agent.player.value,
            "backend": llm_agent.backend,
            "model": llm_agent.model,
            "temperature": llm_agent.temperature,
        },
        "minimax": {
            "player": minimax_agent.player.value,
            "max_depth": getattr(minimax_agent, "max_depth", None),
        },
        "results": {
            "llm_wins": llm_wins,
            "ai_wins": ai_wins,
            "draws_or_unfinished": draws_or_unfinished,
        },
        "game_lengths": game_lengths,
        "avg_game_length": (sum(game_lengths) / len(game_lengths)) if game_lengths else 0.0,
        "games": per_game_results,
        "example_prompt": example_prompt,
    }

    append_jsonl(output_file, summary)
    return summary


if __name__ == "__main__":
    # Przykład: LLM gra wilkiem, Minimax gra owcami
    wolf = LLMAgent(
        player=Player.WOLF,
        #backend="ollama",
        backend="openai",
        #model="llama3.2:3b",
        model="gpt-5.4-nano",
        temperature=0.0,
        verbose=False,
    )

    sheep = MinimaxAgent(
        player=Player.SHEEP,
        max_depth=3,
    )

    summary = run_series(
        num_games=10,
        wolf_agent=wolf,
        sheep_agent=sheep,
        output_file="game_results.jsonl",
        board_size=8,
        max_turns=200,
        verbose_each_game=False,
    )

    print("\n=== PODSUMOWANIE SERII ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nWyniki zostały dopisane do pliku: game_results.jsonl")