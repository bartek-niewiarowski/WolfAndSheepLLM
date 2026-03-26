from __future__ import annotations

from math import inf
from typing import Optional, Tuple
from game import Player, WolfAndSheepGame, Move, Piece

class MinimaxAgent:
    def __init__(self, player: Player, max_depth: int = 4):
        self.player = player
        self.max_depth = max_depth

    def choose_move(self, game: WolfAndSheepGame) -> Optional[Move]:
        legal_moves = game.get_all_valid_moves(game.current_player)
        if not legal_moves:
            return None
        
        _, best_move = self._minimax(game = game, 
            depth = self.max_depth,
            alpha = -inf,
            beta = inf,
            current_player = self.player)

        return best_move

    def _minimax(
            self,
            game: WolfAndSheepGame,
            depth: int,
            alpha: float,
            beta: float,
            current_player: Player,
    ) -> Tuple[float, Optional[Move]]:
        winner = game.get_winner()
        if depth == 0 or winner is not None:
            return self.evaluate(game), None
        
        legal_moves = game.get_all_valid_moves(current_player)

        if not legal_moves:
            return self.evaluate(game), None
        
        maximizing = (current_player == self.player)
        best_move: Optional[Move] = None

        if maximizing:
            best_score = -inf

            for move in legal_moves:
                child = game.successor(move, current_player)
                score, _ = self._minimax(
                    game=child,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    current_player=game.other_player(current_player),
                )

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break

            return best_score, best_move

        else:
            best_score = inf

            for move in legal_moves:
                child = game.successor(move, current_player)
                score, _ = self._minimax(
                    game=child,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    current_player=game.other_player(current_player),
                )

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)
                if beta <= alpha:
                    break

            return best_score, best_move
        
    def evaluate(self, game: WolfAndSheepGame) -> float:
        """
        Zwraca ocenę z perspektywy self.player:
        większa wartość = lepiej dla agenta
        """

        wolf_score = self._evaluate_for_wolf(game)

        if self.player == Player.WOLF:
            return wolf_score
        else:
            return -wolf_score

    def _evaluate_for_wolf(self, game: WolfAndSheepGame) -> float:
        """
        Heurystyka liczona z perspektywy wilka:
        dodatnie = dobrze dla wilka
        ujemne = dobrze dla owiec
        """
        winner = game.get_winner()
        if winner == Player.WOLF:
            return 10000
        if winner == Player.SHEEP:
            return -10000

        wolf_pos = game.find_wolf()
        if wolf_pos is None:
            return -10000

        wolf_row, wolf_col = wolf_pos

        # 1. Postęp wilka w stronę wygranej
        progress_score = (game.board_size - 1 - wolf_row) * 30

        # 2. Mobilność wilka
        wolf_moves = game.get_all_valid_moves(Player.WOLF)
        wolf_mobility_score = len(wolf_moves) * 20

        # 3. Mobilność owiec
        sheep_moves = game.get_all_valid_moves(Player.SHEEP)
        sheep_mobility_penalty = len(sheep_moves) * 4

        # 4. Kara za sąsiednie owce wokół wilka
        adjacent_sheep_penalty = 0
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = wolf_row + dr, wolf_col + dc
            if game.is_within_bounds((nr, nc)):
                if game.get_piece((nr, nc)) == Piece.SHEEP:
                    adjacent_sheep_penalty += 10

        # 5. Premia za bycie bliżej środka planszy
        center = (game.board_size - 1) / 2
        center_distance = abs(wolf_col - center)
        center_score = max(0, 10 - center_distance * 2)

        return (
            progress_score
            + wolf_mobility_score
            + center_score
            - sheep_mobility_penalty
            - adjacent_sheep_penalty
        )