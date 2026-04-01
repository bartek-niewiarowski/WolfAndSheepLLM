from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

Position = Tuple[int, int]

class Piece(Enum):
    EMPTY = 0
    WOLF = 1
    SHEEP = 2

class Player(Enum):
    WOLF = "wolf"
    SHEEP = "sheep"

@dataclass(frozen=True)
class Move:
    start: Position
    end: Position

class InvalidMoveError(Exception):
    pass

class WolfAndSheepGame:
    def __init__(self, board_size: int = 8):
        if board_size < 4 or board_size % 2 != 0:
            raise ValueError("Rozmiar planszy musi być parzysty i >= 4")

        self.board_size = board_size
        self.board: List[List[Piece]] = self._create_initial_board()
        self.current_player: Player = Player.SHEEP
        self.move_history: List[Move] = []

    # =========================
    # Inicjalizacja i kopia
    # =========================

    def _create_initial_board(self) -> List[List[Piece]]:
        board = [
            [Piece.EMPTY for _ in range(self.board_size)]
            for _ in range(self.board_size)
        ]

        # Owce na pierwszym rzędzie, na ciemnych polach
        for col in range(self.board_size):
            if self.is_dark_square((0, col)):
                board[0][col] = Piece.SHEEP

        # Wilk na ostatnim rzędzie, pierwsze ciemne pole
        for col in range(self.board_size):
            if self.is_dark_square((self.board_size - 1, col)):
                board[self.board_size - 1][col] = Piece.WOLF
                break

        return board

    def reset(self) -> None:
        self.board = self._create_initial_board()
        self.current_player = Player.SHEEP
        self.move_history.clear()

    def copy(self) -> "WolfAndSheepGame":
        new_game = WolfAndSheepGame(self.board_size)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history[:]
        return new_game

    # =========================
    # Pomocnicze
    # =========================

    @staticmethod
    def other_player(player: Player) -> Player:
        return Player.WOLF if player == Player.SHEEP else Player.SHEEP

    @staticmethod
    def piece_belongs_to_player(piece: Piece, player: Player) -> bool:
        return (
            (player == Player.WOLF and piece == Piece.WOLF)
            or (player == Player.SHEEP and piece == Piece.SHEEP)
        )

    def is_within_bounds(self, position: Position) -> bool:
        r, c = position
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def is_dark_square(self, position: Position) -> bool:
        r, c = position
        return (r + c) % 2 == 1

    def get_piece(self, position: Position) -> Piece:
        if not self.is_within_bounds(position):
            raise ValueError("Pozycja poza planszą")
        r, c = position
        return self.board[r][c]

    def set_piece(self, position: Position, piece: Piece) -> None:
        if not self.is_within_bounds(position):
            raise ValueError("Pozycja poza planszą")
        r, c = position
        self.board[r][c] = piece

    def find_wolf(self) -> Optional[Position]:
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == Piece.WOLF:
                    return (r, c)
        return None

    def get_piece_positions(self, player: Player) -> List[Position]:
        positions = []
        target_piece = Piece.WOLF if player == Player.WOLF else Piece.SHEEP

        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == target_piece:
                    positions.append((r, c))

        return positions

    # =========================
    # Walidacja ruchu
    # =========================

    def _is_legal_step(self, piece: Piece, start: Position, end: Position) -> bool:
        sr, sc = start
        er, ec = end

        dr = er - sr
        dc = ec - sc

        if abs(dr) != 1 or abs(dc) != 1:
            return False

        if piece == Piece.WOLF:
            return True

        if piece == Piece.SHEEP:
            return dr == 1

        return False

    def is_legal_move_for_player(self, move: Move, player: Player) -> bool:
        start, end = move.start, move.end

        if not (self.is_within_bounds(start) and self.is_within_bounds(end)):
            return False

        if not (self.is_dark_square(start) and self.is_dark_square(end)):
            return False

        piece = self.get_piece(start)

        if piece == Piece.EMPTY:
            return False

        if not self.piece_belongs_to_player(piece, player):
            return False

        if self.get_piece(end) != Piece.EMPTY:
            return False

        return self._is_legal_step(piece, start, end)

    def is_valid_move(self, move: Move) -> bool:
        return self.is_legal_move_for_player(move, self.current_player)

    # =========================
    # Generowanie ruchów
    # =========================

    def _candidate_destinations(self, pos: Position, piece: Piece) -> List[Position]:
        r, c = pos

        if piece == Piece.WOLF:
            return [
                (r - 1, c - 1), (r - 1, c + 1),
                (r + 1, c - 1), (r + 1, c + 1),
            ]

        if piece == Piece.SHEEP:
            return [
                (r + 1, c - 1), (r + 1, c + 1),
            ]

        return []

    def get_valid_moves(self, pos: Position, player: Optional[Player] = None) -> List[Move]:
        player = player or self.current_player

        if not self.is_within_bounds(pos):
            return []

        piece = self.get_piece(pos)
        if piece == Piece.EMPTY:
            return []

        if not self.piece_belongs_to_player(piece, player):
            return []

        moves = []
        for end in self._candidate_destinations(pos, piece):
            move = Move(pos, end)
            if self.is_legal_move_for_player(move, player):
                moves.append(move)

        return moves

    def get_all_valid_moves(self, player: Optional[Player] = None) -> List[Move]:
        player = player or self.current_player
        moves: List[Move] = []

        for pos in self.get_piece_positions(player):
            moves.extend(self.get_valid_moves(pos, player))

        return moves

    # =========================
    # Wykonywanie ruchów
    # =========================

    def make_move(self, move: Move) -> None:
        if not self.is_valid_move(move):
            raise InvalidMoveError(f"Niepoprawny ruch: {move}")

        piece = self.get_piece(move.start)
        self.set_piece(move.start, Piece.EMPTY)
        self.set_piece(move.end, piece)

        self.move_history.append(move)
        self.current_player = self.other_player(self.current_player)

    def apply_move(self, move: Move, player: Optional[Player] = None) -> None:
        player = player or self.current_player

        if not self.is_legal_move_for_player(move, player):
            raise InvalidMoveError(f"Niepoprawny ruch dla gracza {player.value}: {move}")

        piece = self.get_piece(move.start)
        self.set_piece(move.start, Piece.EMPTY)
        self.set_piece(move.end, piece)

        self.move_history.append(move)
        self.current_player = self.other_player(player)

    def successor(self, move: Move, player: Optional[Player] = None) -> "WolfAndSheepGame":
        player = player or self.current_player
        new_state = self.copy()
        new_state.apply_move(move, player)
        return new_state

    # =========================
    # Koniec gry
    # =========================

    def get_winner(self) -> Optional[Player]:
        wolf_pos = self.find_wolf()
        if wolf_pos is None:
            return None

        # Wilk dochodzi do pierwszego rzędu
        if wolf_pos[0] == 0:
            return Player.WOLF

        # Jeśli wilk nie ma ruchów, wygrywają owce
        wolf_moves = self.get_all_valid_moves(Player.WOLF)
        if not wolf_moves:
            return Player.SHEEP

        return None

    def is_game_over(self) -> bool:
        return self.get_winner() is not None

    # =========================
    # Reprezentacja tekstowa / LLM
    # =========================

    def piece_to_char(self, piece: Piece) -> str:
        if piece == Piece.EMPTY:
            return "."
        if piece == Piece.WOLF:
            return "W"
        if piece == Piece.SHEEP:
            return "S"
        return "?"

    def board_as_string(self, show_coordinates: bool = True) -> str:
        lines: List[str] = []

        if show_coordinates:
            header = "   " + " ".join(str(c) for c in range(self.board_size))
            lines.append(header)

        for r in range(self.board_size):
            row_str = " ".join(self.piece_to_char(cell) for cell in self.board[r])
            if show_coordinates:
                lines.append(f"{r:>2} {row_str}")
            else:
                lines.append(row_str)

        return "\n".join(lines)

    def move_to_string(self, move: Move) -> str:
        return f"({move.start[0]},{move.start[1]}) -> ({move.end[0]},{move.end[1]})"

    def legal_moves_as_string(self, player: Optional[Player] = None) -> str:
        player = player or self.current_player
        moves = self.get_all_valid_moves(player)

        if not moves:
            return "No legal moves"

        return "\n".join(
            f"{idx}: {self.move_to_string(move)}"
            for idx, move in enumerate(moves)
        )

    def to_prompt_format(self, player: Optional[Player] = None) -> str:
        player = player or self.current_player
        winner = self.get_winner()

        parts = [
            f"Board size: {self.board_size}x{self.board_size}",
            f"Current player: {player.value}",
            "Board:",
            self.board_as_string(show_coordinates=True),
            "Legal moves:",
            self.legal_moves_as_string(player),
        ]

        if winner is not None:
            parts.append(f"Winner: {winner.value}")

        return "\n".join(parts)

    # =========================
    # Debug
    # =========================

    def __str__(self) -> str:
        return self.board_as_string(show_coordinates=True)