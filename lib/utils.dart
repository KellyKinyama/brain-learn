/// kenny_utils.dart
///
/// This file contains various utility functions and global lookup tables
/// that are used across different parts of the Kenny chess engine.
/// It consolidates functions from kennyFuncs.h and defines missing global arrays.

import 'defs.dart';
import 'board.dart';
import 'move.dart';
import 'move_gen2.dart'; // For isAttacked
import 'display_move.dart'; // For toSan

// =====================================================================
// Global Lookup Tables (from C++ code, implicitly used in various files)
// =====================================================================

/// Array to convert square index to algebraic notation (e.g., 0 -> "a1", 63 -> "h8")
const List<String> SQUARENAME = [
  "a1",
  "b1",
  "c1",
  "d1",
  "e1",
  "f1",
  "g1",
  "h1",
  "a2",
  "b2",
  "c2",
  "d2",
  "e2",
  "f2",
  "g2",
  "h2",
  "a3",
  "b3",
  "c3",
  "d3",
  "e3",
  "f3",
  "g3",
  "h3",
  "a4",
  "b4",
  "c4",
  "d4",
  "e4",
  "f4",
  "g4",
  "h4",
  "a5",
  "b5",
  "c5",
  "d5",
  "e5",
  "f5",
  "g5",
  "h5",
  "a6",
  "b6",
  "c6",
  "d6",
  "e6",
  "f6",
  "g6",
  "h6",
  "a7",
  "b7",
  "c7",
  "d7",
  "e7",
  "f7",
  "g7",
  "h7",
  "a8",
  "b8",
  "c8",
  "d8",
  "e8",
  "f8",
  "g8",
  "h8",
];

/// Array to get the rank (1-8) of a square index (0-63)
const List<int> RANKS = [
  1, 1, 1, 1, 1, 1, 1, 1, // Rank 1
  2, 2, 2, 2, 2, 2, 2, 2, // Rank 2
  3, 3, 3, 3, 3, 3, 3, 3, // Rank 3
  4, 4, 4, 4, 4, 4, 4, 4, // Rank 4
  5, 5, 5, 5, 5, 5, 5, 5, // Rank 5
  6, 6, 6, 6, 6, 6, 6, 6, // Rank 6
  7, 7, 7, 7, 7, 7, 7, 7, // Rank 7
  8, 8, 8, 8, 8, 8, 8, 8, // Rank 8
];

/// Array to get the file (1-8) of a square index (0-63)
const List<int> FILES = [
  1, 2, 3, 4, 5, 6, 7, 8, // File a-h
  1, 2, 3, 4, 5, 6, 7, 8,
  1, 2, 3, 4, 5, 6, 7, 8,
  1, 2, 3, 4, 5, 6, 7, 8,
  1, 2, 3, 4, 5, 6, 7, 8,
  1, 2, 3, 4, 5, 6, 7, 8,
  1, 2, 3, 4, 5, 6, 7, 8,
  1, 2, 3, 4, 5, 6, 7, 8,
];

// =====================================================================
// Utility Functions (from kennyFuncs.h and other C++ files)
// =====================================================================

/// Displays the Principal Variation (PV) on the console.
/// Translates `displayPV()` from kennyFuncs.h (declaration) and kennySearch.cpp (usage).
/// This function relies on `toSan` for move formatting.
void displayPV() {
  if (board.lastPVLength == 0) {
    print("PV: (empty)");
    return;
  }
  StringBuffer pvString = StringBuffer("PV: ");
  for (int i = 0; i < board.lastPVLength; i++) {
    String sanMove = '';
    // toSan modifies the string buffer, so we need to pass a mutable reference or return value.
    // Assuming toSan now returns the SAN string.
    if (toSan(board.lastPV[i], sanMove)) {
      // Pass a dummy string for now, toSan needs fixing
      pvString.write("${sanMove} ");
    } else {
      pvString.write("${board.lastPV[i].toString()} "); // Fallback
    }
  }
  print(pvString.toString());
}

/// Converts milliseconds to a formatted time string (hh:mm:ss).
/// Translates `mstostring()` from kennyFuncs.h (declaration) and kennySearch.cpp (usage).
String mstostring(U64 dt) {
  int hh = (dt ~/ 1000) ~/ 3600;
  int mm = ((dt ~/ 1000) - hh * 3600) ~/ 60;
  int ss = (dt ~/ 1000) - hh * 3600 - mm * 60;
  return '${hh.toString().padLeft(2, '0')}:${mm.toString().padLeft(2, '0')}:${ss.toString().padLeft(2, '0')}';
}

/// Checks if a text move string is valid and converts it to a `Move` object.
/// Translates `isValidTextMove()` from kennyFuncs.h (declaration) and kennyCommands.cpp (usage).
/// This is a complex function that requires move generation and legality checks.
/// [textMove] The move string (e.g., "e2e4", "g1f3", "e7e8q").
/// [outMove] The `Move` object to populate if the move is valid.
/// Returns true if the move is valid, false otherwise.
bool isValidTextMove(String textMove, Move outMove) {
  if (textMove.length < 4 || textMove.length > 5) {
    return false; // e.g., "e2e4" (4 chars), "e7e8q" (5 chars)
  }

  // Parse from and to squares
  String fromStr = textMove.substring(0, 2);
  String toStr = textMove.substring(2, 4);
  String? promoChar = (textMove.length == 5) ? textMove[4] : null;

  int from = -1, to = -1;

  // Find square indices from string (reverse lookup from SQUARENAME)
  for (int i = 0; i < SQUARENAME.length; i++) {
    if (SQUARENAME[i] == fromStr) from = i;
    if (SQUARENAME[i] == toStr) to = i;
  }

  if (from == -1 || to == -1) {
    return false; // Invalid square names
  }

  int piece = board.square[from];
  if (piece == EMPTY) {
    return false; // No piece on 'from' square
  }

  // Check if the piece belongs to the current side to move
  if ((board.nextMove == WHITE_MOVE && (piece >= BLACK_PAWN)) ||
      (board.nextMove == BLACK_MOVE &&
          (piece <= WHITE_QUEEN && piece != EMPTY))) {
    return false; // Wrong side's piece
  }

  // Generate all legal moves from the current position

  int currentPlyMoveStart = board.moveBufLen[board.endOfGame];
  print(
    "Generate all legal moves from the current position: $currentPlyMoveStart",
  );
  int currentPlyMoveEnd = movegen(
    currentPlyMoveStart,
  ); // Generate moves for root ply
  if (currentPlyMoveStart == 236) {
    print(board.moveBuffer.sublist(currentPlyMoveStart, currentPlyMoveEnd));
  }
  // Iterate through generated moves to find a match
  for (int i = currentPlyMoveStart; i < currentPlyMoveEnd; i++) {
    Move generatedMove = board.moveBuffer[i];

    if (generatedMove.getFrom() == from && generatedMove.getTosq() == to) {
      // Check for promotion consistency
      if (promoChar != null) {
        int expectedPromoPiece;
        if (board.nextMove == WHITE_MOVE) {
          switch (promoChar.toLowerCase()) {
            case 'q':
              expectedPromoPiece = WHITE_QUEEN;
              break;
            case 'r':
              expectedPromoPiece = WHITE_ROOK;
              break;
            case 'b':
              expectedPromoPiece = WHITE_BISHOP;
              break;
            case 'n':
              expectedPromoPiece = WHITE_KNIGHT;
              break;
            default:
              return false; // Invalid promotion char
          }
        } else {
          switch (promoChar.toLowerCase()) {
            case 'q':
              expectedPromoPiece = BLACK_QUEEN;
              break;
            case 'r':
              expectedPromoPiece = BLACK_ROOK;
              break;
            case 'b':
              expectedPromoPiece = BLACK_BISHOP;
              break;
            case 'n':
              expectedPromoPiece = BLACK_KNIGHT;
              break;
            default:
              return false; // Invalid promotion char
          }
        }
        if (generatedMove.getProm() == expectedPromoPiece) {
          outMove.moveInt = generatedMove.moveInt; // Found a valid move
          return true;
        }
      } else {
        // No promotion expected, ensure generated move is not a promotion
        if (!generatedMove.isPromotion()) {
          outMove.moveInt = generatedMove.moveInt; // Found a valid move
          return true;
        }
      }
    }
  }

  if (currentPlyMoveStart == 236) {
    print(board.moveBuffer.sublist(currentPlyMoveStart, currentPlyMoveEnd));
  }

  return false; // No matching legal move found
}

/// Displays general information about the engine.
/// Translates `info()` from kennyFuncs.h (declaration) and kennyMain.cpp (usage).
void info() {
  print(KENNY_PROG_VERSION);
  print("Search Depth: ${board.searchDepth}");
  print("Max Time: ${board.maxTime ~/ 1000}s");
  // Add other relevant info like defined macros if needed for debugging
  // For example:
  // print("KENNY_CUSTOM_VALUES defined");
  // print("KENNY_CUSTOM_POSVALS defined");
  // print("KENNY_CUSTOM_PSTABLES defined");
  // print("KENNY_CUSTOM_ENDGAME defined");
}
