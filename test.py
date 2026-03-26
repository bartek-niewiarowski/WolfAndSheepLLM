from typing import Optional

from game import WolfAndSheepGame, Player
from minimaxAgent import MinimaxAgent
from llmAgent import LLMAgent

def play_game(
    wolf_agent: MinimaxAgent,
    sheep_agent: LLMAgent,
    board_size: int = 8,
    max_turns: int = 200,
) -> tuple[Optional[Player], WolfAndSheepGame]:

    game = WolfAndSheepGame(board_size=board_size)

    print("=== POCZĄTKOWA PLANSZA ===")
    print(game)
    print()

    for turn in range(max_turns):
        if game.is_game_over():
            break

        if game.current_player == Player.WOLF:
            move = wolf_agent.choose_move(game)
        else:
            move = sheep_agent.choose_move(game)

        if move is None:
            break

        game.make_move(move)

    print("=== KOŃCOWA PLANSZA ===")
    print(game)
    print()

    return game.get_winner(), game


if __name__ == "__main__":
    sheep = MinimaxAgent(player=Player.SHEEP, max_depth=1)
    wolf = LLMAgent(player=Player.WOLF, project="", verbose=True)

    winner, game = play_game(wolf_agent=wolf, sheep_agent=sheep)

    if winner:
        print(f"Zwycięzca: {winner.value}")
    else:
        print("Remis lub przekroczono limit ruchów")