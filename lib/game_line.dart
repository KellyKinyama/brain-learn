/// kenny_game_line.dart
///
/// This file defines the `GameLineRecord` class, which stores information
/// about a specific position in the game line, including the move that led to it,
/// castling rights, en passant square, fifty-move rule counter, and hash key.
/// It translates the C++ `GameLineRecord` struct from kennyGameLine.h.

import 'defs.dart';
import 'move.dart';

class GameLineRecord {
  Move move; // The move that led to this position
  int castleWhite; // White's castle status (CANCASTLEOO, CANCASTLEOOO)
  int castleBlack; // Black's castle status (CANCASTLEOO, CANCASTLEOOO)
  int epSquare; // En-passant target square after double pawn move
  int fiftyMove; // Moves since the last pawn move or capture
  U64 key; // Hash key of the position

  /// Constructor for GameLineRecord.
  GameLineRecord({
    required this.move,
    this.castleWhite = 0,
    this.castleBlack = 0,
    this.epSquare = 0,
    this.fiftyMove = 0,
    this.key = 0,
  });

  /// Factory constructor to create an empty record.
  factory GameLineRecord.empty() {
    return GameLineRecord(move: NOMOVE);
  }
}
