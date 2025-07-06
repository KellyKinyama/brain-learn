/// kenny_move.dart
///
/// This file defines the `Move` class, which represents a chess move.
/// It translates the C++ `Move` struct from kennyMove.h and its methods from kennyMove.cpp.
/// The move is stored as a single integer (`moveInt`) with bitwise operations
/// used to encode and decode move properties.

import 'defs.dart';
import 'utils.dart'; // Import necessary definitions

class Move {
  U64 moveInt; // The integer representation of the move

  /// Constructor to initialize a move.
  Move({this.moveInt = 0});

  /// Clears the move, setting its integer representation to 0.
  void clear() {
    moveInt = 0;
  }

  /// Sets the 'from' square (bits 0-5).
  void setFrom(int from) {
    moveInt &= 0xFFFFFFC0; // Clear bits 0-5
    moveInt |= (from & 0x0000003F); // Set bits 0-5
  }

  /// Sets the 'to' square (bits 6-11).
  void setTosq(int tosq) {
    moveInt &= 0xFFFFE03F; // Clear bits 6-11
    moveInt |= (tosq & 0x0000003F) << 6; // Set bits 6-11
  }

  /// Sets the piece that moves (bits 12-15).
  void setPiec(int piec) {
    moveInt &= 0xFFFF0FFF; // Clear bits 12-15
    moveInt |= (piec & 0x0000000F) << 12; // Set bits 12-15
  }

  /// Sets the captured piece (bits 16-19).
  void setCapt(int capt) {
    moveInt &= 0xFFF0FFFF; // Clear bits 16-19
    moveInt |= (capt & 0x0000000F) << 16; // Set bits 16-19
  }

  /// Sets the promotion piece (bits 20-23).
  void setProm(int prom) {
    moveInt &= 0xFF0FFFFF; // Clear bits 20-23
    moveInt |= (prom & 0x0000000F) << 20; // Set bits 20-23
  }

  /// Sets the castling flag (bit 24).
  void setCastle(bool castle) {
    if (castle) {
      moveInt |= 0x01000000; // Set bit 24
    } else {
      moveInt &= ~0x01000000; // Clear bit 24
    }
  }

  /// Sets the en passant flag (bit 25).
  void setEnpassant(bool enpassant) {
    if (enpassant) {
      moveInt |= 0x02000000; // Set bit 25
    } else {
      moveInt &= ~0x02000000; // Clear bit 25
    }
  }

  /// Sets the pawn double move flag (bit 26).
  void setPawnDoubleMove(bool pawnDoubleMove) {
    if (pawnDoubleMove) {
      moveInt |= 0x04000000; // Set bit 26
    } else {
      moveInt &= ~0x04000000; // Clear bit 26
    }
  }

  /// Gets the 'from' square (bits 0-5).
  int getFrom() {
    return (moveInt & 0x0000003F);
  }

  /// Gets the 'to' square (bits 6-11).
  int getTosq() {
    return ((moveInt & 0x00000FC0) >> 6);
  }

  /// Gets the piece that moves (bits 12-15).
  int getPiec() {
    return ((moveInt & 0x0000F000) >> 12);
  }

  /// Gets the captured piece (bits 16-19).
  int getCapt() {
    return ((moveInt & 0x000F0000) >> 16);
  }

  /// Gets the promotion piece (bits 20-23).
  int getProm() {
    return ((moveInt & 0x00F00000) >> 20);
  }

  /// Checks if the move is a castling move (bit 24).
  bool isCastle() {
    return (moveInt & 0x01000000) != 0;
  }

  /// Checks if the move is an en passant capture (bit 25).
  bool isEnpassant() {
    return (moveInt & 0x02000000) != 0;
  }

  /// Checks if the move is a pawn double move (bit 26).
  bool isPawnDoublemove() {
    // This logic is more complex in the C++ code, checking ranks.
    // For now, it's a direct translation of the flag.
    // The C++ code also checks ranks of from and to squares.
    // This simplification assumes the flag is set correctly.
    return (moveInt & 0x04000000) != 0;
  }

  /// Checks if the move is a capture (captured piece is not EMPTY).
  bool isCapture() {
    return getCapt() != EMPTY;
  }

  /// Checks if the move is a promotion (promotion piece is not EMPTY).
  bool isPromotion() {
    return getProm() != EMPTY;
  }

  /// Checks if the move is a king-side castling (O-O).
  /// This checks if the piece is a king and the move is a castle,
  /// and then specifically checks the 'from' and 'to' squares for O-O.
  bool isCastleOO() {
    // Assuming WHITE_KING or BLACK_KING is the moving piece.
    // E1 to G1 for white, E8 to G8 for black.
    if (!isCastle()) return false;
    int piece = getPiec();
    int from = getFrom();
    int to = getTosq();

    if (piece == WHITE_KING) {
      return from == E1 && to == G1;
    } else if (piece == BLACK_KING) {
      return from == E8 && to == G8;
    }
    return false;
  }

  /// Checks if the move is a queen-side castling (O-O-O).
  /// This checks if the piece is a king and the move is a castle,
  /// and then specifically checks the 'from' and 'to' squares for O-O-O.
  bool isCastleOOO() {
    // Assuming WHITE_KING or BLACK_KING is the moving piece.
    // E1 to C1 for white, E8 to C8 for black.
    if (!isCastle()) return false;
    int piece = getPiec();
    int from = getFrom();
    int to = getTosq();

    if (piece == WHITE_KING) {
      return from == E1 && to == C1;
    } else if (piece == BLACK_KING) {
      return from == E8 && to == C8;
    }
    return false;
  }

  /// Checks if the move is a pawn move (piece is a pawn).
  bool isPawnmove() {
    int piece = getPiec();
    return piece == WHITE_PAWN || piece == BLACK_PAWN;
  }

  /// Overriding equality operator for Move objects.
  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Move && moveInt == other.moveInt;
  }

  /// Overriding hashCode for Move objects.
  @override
  int get hashCode => moveInt.hashCode;

  /// For debugging or display purposes.
  @override
  String toString() {
    // A simplified toString for now. A full SAN conversion would be complex.
    String fromSq = SQUARENAME[getFrom()];
    String toSq = SQUARENAME[getTosq()];
    String pieceChar = PIECECHARS[getPiec()];
    String capturedChar = isCapture() ? 'x${PIECECHARS[getCapt()]}' : '';
    String promotionChar = isPromotion() ? '=${PIECECHARS[getProm()]}' : '';

    if (isCastleOO()) return 'O-O';
    if (isCastleOOO()) return 'O-O-O';

    return '$pieceChar$fromSq$capturedChar$toSq$promotionChar';
  }

  String toAlgebraic() {
    // A simplified toString for now. A full SAN conversion would be complex.
    String pieceChar = PIECECHARS[getPiec()];
    String fromSq = SQUARENAME[getFrom()];
    String toSq = SQUARENAME[getTosq()];
    String promotionChar =
        isPromotion() && !(pieceChar == 'K' || pieceChar == 'k')
        ? '=${PIECECHARS[getProm()]}'
        : '';

    if (isCastleOO()) return 'O-O';
    if (isCastleOOO()) return 'O-O-O';

    return '$fromSq$toSq$promotionChar';
  }
}

// Global NOMOVE instance (from kennyGlobals.h)
final Move NOMOVE = Move();
