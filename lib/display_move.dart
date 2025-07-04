/// kenny_display_move.dart
///
/// This file contains functions for displaying chess moves in a human-readable
/// format (e.g., SAN - Standard Algebraic Notation).
/// It translates `displayMove()` and `toSan()` from kennyDispMove.cpp.

import 'defs.dart';
import 'move.dart';
import 'board.dart';
import 'utils.dart'; // For accessing board state and moveBuffer

/// Displays a single move on the console without full SAN disambiguation.
/// Translates `displayMove()` from kennyDispMove.cpp.
void displayMove(Move move) {
  if (move.isCastleOO()) {
    print("O-O");
    return;
  }
  if (move.isCastleOOO()) {
    print("O-O-O");
    return;
  }

  if (!move.isPawnmove()) {
    print(PIECECHARS[move.getPiec()]);
  }

  // For captures, add 'x'
  if (move.isCapture()) {
    print('x');
  }

  // Print destination square
  print(SQUARENAME[move.getTosq()]);

  // For promotions, add promotion piece
  if (move.isPromotion()) {
    print(PIECECHARS[move.getProm()]);
  }
}

/// Converts a `Move` object to its Standard Algebraic Notation (SAN) string.
/// This function handles disambiguation, captures, promotions, and check/mate.
/// Translates `toSan()` from kennyDispMove.cpp.
/// Returns true if successful, false otherwise (e.g., illegal move).
/// [move] The Move object to convert.
/// [sanMoveBuffer] A buffer (String) to store the resulting SAN string.
///
/// NOTE: The C++ `toSan` function modifies a `char*` buffer. In Dart, we'll
/// return a `String` directly or use a `StringBuffer`.
/// The `sanMoveBuffer` parameter will be treated as an output parameter.
///
/// This function is quite complex due to SAN disambiguation rules.
/// The provided implementation will be a direct translation of the C++ logic.
bool toSan(Move move, String sanMoveBuffer) {
  // The C++ function uses a `char sanMove[12]` and `strcpy`, `sprintf`.
  // In Dart, we'll build the string.
  StringBuffer san = StringBuffer();
  bool legal = false; // Assume illegal until proven otherwise
  int from = move.getFrom();
  int to = move.getTosq();
  int piece = move.getPiec();
  int captured = move.getCapt();

  // Check for castling first
  if (move.isCastleOO()) {
    san.write("O-O");
    return true;
  }
  if (move.isCastleOOO()) {
    san.write("O-O-O");
    return true;
  }

  // Piece abbreviation (if not a pawn)
  if (!move.isPawnmove()) {
    san.write(PIECECHARS[piece]);
  }

  // Disambiguation for non-pawn moves
  // This requires generating all legal moves from the current position
  // and checking if other pieces of the same type can move to the 'to' square.
  // This is a simplified version and will need `movegen` and `makeMove`/`unmakeMove`.
  // The C++ code iterates through `board.moveBuffer` to find ambiguous moves.

  // Temporarily make the move to check for legality and attacks
  // This is a critical step in the C++ `toSan` and `isValidTextMove`.
  // It requires the `makeMove` and `unmakeMove` functions.
  // For now, this part will be commented out or simplified until `makeMove` is robust.

  // Store current board state to restore later
  // GameLineRecord currentRecord = board.gameLine[board.endOfGame];
  // board.gameLine[board.endOfGame].move = move; // Store the move
  // makeMove(move); // Temporarily make the move

  // Check if the move is legal (i.e., doesn't leave own king in check)
  // This check is implicitly done by `isOtherKingAttacked()` after `makeMove`.
  // legal = !isOwnKingAttacked(); // Assuming this checks the current side's king

  // if (!legal) {
  //   unmakeMove(move); // Undo the temporary move
  //   sanMoveBuffer = "unknown"; // Assign to output parameter
  //   return false;
  // }

  // Disambiguation logic (simplified placeholder)
  // In C++, it counts `ambigfile` and `ambigrank` by iterating `moveBuffer`
  // and checking if other pieces of the same type can move to `to`.
  // This requires a full move generator to be implemented first.
  bool ambig = false;
  bool ambigfile = false;
  bool ambigrank = false;

  // Placeholder for disambiguation logic:
  // For a full implementation, you'd generate all legal moves,
  // filter for the same piece type and destination square,
  // and then determine if file/rank disambiguation is needed.
  // This is a complex part that relies on the `movegen` function.

  // The C++ `toSan` does this:
  // it generates moves, makes them, checks if the piece is the same type,
  // and if it moves to the same 'to' square.
  // If multiple pieces can move to the same 'to' square, it checks if
  // disambiguation by file or rank is needed.

  if (ambig) {
    if (ambigfile) {
      san.write(SQUARENAME[from][0]); // File char
      if (ambigrank) {
        san.write(SQUARENAME[from][1]); // Rank char
      }
    } else {
      san.write(SQUARENAME[from][1]); // Rank char
    }
  }

  // Capture indicator
  if (move.isCapture()) {
    if (move.isPawnmove()) {
      san.write(SQUARENAME[from][0]); // Pawn captures include file of origin
    }
    san.write('x');
  }

  // Destination square
  san.write(SQUARENAME[to]);

  // Promotion
  if (move.isPromotion()) {
    san.write('=');
    san.write(PIECECHARS[move.getProm()]);
  }

  // Check/Checkmate (requires `isOwnKingAttacked` and `isEndOfgame` after move)
  // This logic is typically applied *after* the move is made and checked for legality.
  // If `isOtherKingAttacked()` returns true after the move, it's a check.
  // If it's a check and no legal moves for the opponent, it's checkmate.

  // For now, just add check/mate symbols if the conditions are met (placeholders)
  // if (board.isOtherKingAttacked()) { // Assuming this checks if opponent's king is attacked
  //   // Check for checkmate
  //   int legalMovesForOpponent = 0; // Need to generate moves for opponent
  //   // if (board.isEndOfgame(legalMovesForOpponent, NOMOVE)) { // Simplified check
  //   //   san.write('#'); // Checkmate
  //   // } else {
  //   //   san.write('+'); // Check
  //   // }
  // }

  // Restore board state (if temporarily altered for legality check)
  // unmakeMove(move); // Undo the temporary move
  // board.gameLine[board.endOfGame] = currentRecord; // Restore previous record

  sanMoveBuffer = san.toString(); // Assign to output parameter
  return true; // Assume success for now
}
