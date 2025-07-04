/// kenny_read_fen.dart
///
/// This file contains functions for reading and parsing FEN (Forsyth-Edwards Notation) strings.
/// It translates `readFen()` from kennyReadFen.cpp and `setupFen()` from kennySetup.cpp.
/// Since Dart doesn't directly interact with file systems in a browser environment,
/// `readFen` will be adapted to `readFenString` for direct string input.

import 'defs.dart';
import 'board.dart';

/// Parses a FEN string and initializes the board state.
/// This function combines the logic of `readFen` (for parsing the string)
/// and `setupFen` (for applying the parsed state to the board).
/// Returns true if the FEN string is valid and the board is set, false otherwise.
bool readFenString(String fen) {
  List<String> parts = fen.split(' ');
  if (parts.length < 4 || parts.length > 6) {
    print(
      "Error: Invalid FEN string format. Expected 4-6 parts, got ${parts.length}.",
    );
    return false;
  }

  String piecePlacement = parts[0];
  String activeColor = parts[1];
  String castlingRights = parts[2];
  String enPassantTarget = parts[3];
  int halfmoveClock = (parts.length > 4) ? int.tryParse(parts[4]) ?? 0 : 0;
  int fullmoveNumber = (parts.length > 5) ? int.tryParse(parts[5]) ?? 1 : 1;

  List<int> tempSquares = List.filled(64, EMPTY);
  int file = 1;
  int rank = 8; // FEN starts from rank 8

  // Parse piece placement
  for (int i = 0; i < piecePlacement.length; i++) {
    String char = piecePlacement[i];
    if (char == '/') {
      rank--;
      file = 1;
    } else if (int.tryParse(char) != null) {
      int emptySquares = int.parse(char);
      file += emptySquares; // Skip empty squares
    } else {
      int pieceType;
      switch (char) {
        case 'P':
          pieceType = WHITE_PAWN;
          break;
        case 'N':
          pieceType = WHITE_KNIGHT;
          break;
        case 'B':
          pieceType = WHITE_BISHOP;
          break;
        case 'R':
          pieceType = WHITE_ROOK;
          break;
        case 'Q':
          pieceType = WHITE_QUEEN;
          break;
        case 'K':
          pieceType = WHITE_KING;
          break;
        case 'p':
          pieceType = BLACK_PAWN;
          break;
        case 'n':
          pieceType = BLACK_KNIGHT;
          break;
        case 'b':
          pieceType = BLACK_BISHOP;
          break;
        case 'r':
          pieceType = BLACK_ROOK;
          break;
        case 'q':
          pieceType = BLACK_QUEEN;
          break;
        case 'k':
          pieceType = BLACK_KING;
          break;
        default:
          print("Error: Invalid piece character in FEN: $char");
          return false;
      }
      tempSquares[BOARDINDEX[file][rank]] = pieceType;
      file++;
    }
  }

  // Parse active color
  int nextMove;
  if (activeColor == 'w') {
    nextMove = WHITE_MOVE;
  } else if (activeColor == 'b') {
    nextMove = BLACK_MOVE;
  } else {
    print("Error: Invalid active color in FEN: $activeColor");
    return false;
  }

  // Parse castling rights
  int whiteCastle = 0;
  int blackCastle = 0;
  if (castlingRights.contains('K')) whiteCastle |= CANCASTLEOO;
  if (castlingRights.contains('Q')) whiteCastle |= CANCASTLEOOO;
  if (castlingRights.contains('k')) blackCastle |= CANCASTLEOO;
  if (castlingRights.contains('q')) blackCastle |= CANCASTLEOOO;

  // Parse en passant target square
  int epSq = 0;
  if (enPassantTarget != '-') {
    if (enPassantTarget.length != 2) {
      print("Error: Invalid en passant target square format: $enPassantTarget");
      return false;
    }
    int epFile = enPassantTarget.codeUnitAt(0) - 'a'.codeUnitAt(0) + 1;
    int epRank = int.tryParse(enPassantTarget[1]) ?? 0;
    if (epFile < 1 || epFile > 8 || epRank < 1 || epRank > 8) {
      print("Error: Invalid en passant target square: $enPassantTarget");
      return false;
    }
    epSq = BOARDINDEX[epFile][epRank];
  }

  // Initialize the board with the parsed values
  board.initFromSquares(
    tempSquares,
    nextMove,
    halfmoveClock,
    whiteCastle,
    blackCastle,
    epSq,
  );

  // Note: fullmoveNumber is typically used for game tracking, not direct board state.
  // The C++ code uses it, but it's not directly part of `initFromSquares`.

  return true;
}
