/// kenny_make_move.dart
///
/// This file contains the implementation for `makeMove` and `unmakeMove` functions.
/// These functions are crucial for updating the board state after a move,
/// including piece positions, material, castling rights, en passant square,
/// fifty-move rule, and Zobrist hash key.
/// They translate the logic from kennyMakeMove.cpp.

import 'defs.dart';
import 'board.dart';
import 'game_line.dart';
import 'move.dart';
import 'hash.dart';
import 'bit_ops.dart'; // For bitCnt

/// Applies a given move to the board, updating all relevant state variables.
/// Translates `makeMove()` from kennyMakeMove.cpp.
void makeMove(Move move) {
  int from = move.getFrom();
  int to = move.getTosq();
  int piece = move.getPiec();
  int captured = move.getCapt();
  int promotion = move.getProm();

  // Store current board state in gameLine before making the move
  board.gameLine[board.endOfGame].move = move;
  board.gameLine[board.endOfGame].castleWhite = board.castleWhite;
  board.gameLine[board.endOfGame].castleBlack = board.castleBlack;
  board.gameLine[board.endOfGame].fiftyMove = board.fiftyMove;
  board.gameLine[board.endOfGame].epSquare = board.epSquare;
  board.gameLine[board.endOfGame].key = board.hashkey;

  board.endOfGame++;
  board.endOfSearch++;

  // Update fifty-move rule counter
  board.fiftyMove++;
  if (move.isPawnmove() || move.isCapture()) {
    board.fiftyMove = 0;
  }

  // Update en passant square hash key
  if (board.epSquare != 0) {
    board.hashkey ^= KEY.ep[board.epSquare];
  }
  board.epSquare = 0; // Reset en passant square

  // --- Start of Castling Rights Update ---
  // This logic is critical and mirrors the C++ implementation carefully.
  // Rights are revoked if the king moves, a rook moves from its home square,
  // or a rook is captured on its home square.

  // 1. Update based on the piece that MOVES
  if (piece == WHITE_KING) {
    if ((board.castleWhite & CANCASTLEOO) != 0) board.hashkey ^= KEY.wk;
    if ((board.castleWhite & CANCASTLEOOO) != 0) board.hashkey ^= KEY.wq;
    board.castleWhite = 0;
  } else if (piece == BLACK_KING) {
    if ((board.castleBlack & CANCASTLEOO) != 0) board.hashkey ^= KEY.bk;
    if ((board.castleBlack & CANCASTLEOOO) != 0) board.hashkey ^= KEY.bq;
    board.castleBlack = 0;
  }

  if (from == A1) {
    if ((board.castleWhite & CANCASTLEOOO) != 0) {
      board.hashkey ^= KEY.wq;
      board.castleWhite &= ~CANCASTLEOOO;
    }
  } else if (from == H1) {
    if ((board.castleWhite & CANCASTLEOO) != 0) {
      board.hashkey ^= KEY.wk;
      board.castleWhite &= ~CANCASTLEOO;
    }
  } else if (from == A8) {
    if ((board.castleBlack & CANCASTLEOOO) != 0) {
      board.hashkey ^= KEY.bq;
      board.castleBlack &= ~CANCASTLEOOO;
    }
  } else if (from == H8) {
    if ((board.castleBlack & CANCASTLEOO) != 0) {
      board.hashkey ^= KEY.bk;
      board.castleBlack &= ~CANCASTLEOO;
    }
  }

  // 2. Update based on a CAPTURE on a rook's starting square
  if (captured != EMPTY) {
    if (to == A1) {
      if ((board.castleWhite & CANCASTLEOOO) != 0) {
        board.hashkey ^= KEY.wq;
        board.castleWhite &= ~CANCASTLEOOO;
      }
    } else if (to == H1) {
      if ((board.castleWhite & CANCASTLEOO) != 0) {
        board.hashkey ^= KEY.wk;
        board.castleWhite &= ~CANCASTLEOO;
      }
    } else if (to == A8) {
      if ((board.castleBlack & CANCASTLEOOO) != 0) {
        board.hashkey ^= KEY.bq;
        board.castleBlack &= ~CANCASTLEOOO;
      }
    } else if (to == H8) {
      if ((board.castleBlack & CANCASTLEOO) != 0) {
        board.hashkey ^= KEY.bk;
        board.castleBlack &= ~CANCASTLEOO;
      }
    }
  }
  // --- End of Castling Rights Update ---

  // Update hash key for side to move
  board.hashkey ^= KEY.side;

  // Update bitboards and square array
  BitMap fromToBitMap = BITSET[from] | BITSET[to];

  // Remove piece from 'from' square
  board.hashkey ^= KEY.keys[from][piece];
  board.square[from] = EMPTY;

  // Handle captures
  if (captured != EMPTY) {
    if (!move.isEnpassant()) {
      board.hashkey ^= KEY.keys[to][captured];
      _updateMaterialAndBitboardsForCapture(captured, to);
    }
  }

  // Place piece on 'to' square
  board.hashkey ^= KEY.keys[to][piece];
  board.square[to] = piece;

  // Handle specific move types
  if (move.isPawnDoublemove()) {
    board.epSquare = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.hashkey ^= KEY.ep[board.epSquare];
  } else if (move.isEnpassant()) {
    int capturedPawnSq = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.square[capturedPawnSq] = EMPTY;
    board.hashkey ^= KEY.keys[capturedPawnSq][captured];
    _updateMaterialAndBitboardsForCapture(captured, capturedPawnSq);
  } else if (move.isCastle()) {
    if (to == G1) {
      // White O-O
      board.square[H1] = EMPTY;
      board.square[F1] = WHITE_ROOK;
      board.hashkey ^= KEY.keys[H1][WHITE_ROOK] ^ KEY.keys[F1][WHITE_ROOK];
    } else if (to == C1) {
      // White O-O-O
      board.square[A1] = EMPTY;
      board.square[D1] = WHITE_ROOK;
      board.hashkey ^= KEY.keys[A1][WHITE_ROOK] ^ KEY.keys[D1][WHITE_ROOK];
    } else if (to == G8) {
      // Black O-O
      board.square[H8] = EMPTY;
      board.square[F8] = BLACK_ROOK;
      board.hashkey ^= KEY.keys[H8][BLACK_ROOK] ^ KEY.keys[F8][BLACK_ROOK];
    } else if (to == C8) {
      // Black O-O-O
      board.square[A8] = EMPTY;
      board.square[D8] = BLACK_ROOK;
      board.hashkey ^= KEY.keys[A8][BLACK_ROOK] ^ KEY.keys[D8][BLACK_ROOK];
    }
  } else if (move.isPromotion()) {
    board.hashkey ^= KEY.keys[to][piece]; // XOR out pawn
    board.hashkey ^= KEY.keys[to][promotion]; // XOR in promoted piece
    board.square[to] = promotion;
    _updateMaterialForPromotion(piece, promotion);
  }

  // Rebuild all bitboards from the square array for consistency
  _rebuildBitboards();

  // Toggle side to move
  board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;
}

/// Undoes the last move, restoring the board to its previous state.
/// Translates `unmakeMove()` from kennyMakeMove.cpp.
void unmakeMove(Move move) {
  board.endOfGame--;
  board.endOfSearch--;

  // Restore state from gameLine, which is the most reliable way
  GameLineRecord prevRecord = board.gameLine[board.endOfGame];
  board.castleWhite = prevRecord.castleWhite;
  board.castleBlack = prevRecord.castleBlack;
  board.epSquare = prevRecord.epSquare;
  board.fiftyMove = prevRecord.fiftyMove;
  board.hashkey = prevRecord.key; // Restore hash key completely

  // Toggle side to move back
  board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;

  // Since we restored the hash key, we just need to restore the board arrays
  int from = move.getFrom();
  int to = move.getTosq();
  int piece = move.getPiec();
  int captured = move.getCapt();
  int promotion = move.getProm();

  // Move piece back
  board.square[from] = piece;

  // Handle special move types for unmaking
  if (move.isEnpassant()) {
    board.square[to] = EMPTY;
    int capturedPawnSq = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.square[capturedPawnSq] = captured;
  } else if (move.isCastle()) {
    board.square[to] = EMPTY;
    if (to == G1) {
      // White O-O
      board.square[H1] = WHITE_ROOK;
      board.square[F1] = EMPTY;
    } else if (to == C1) {
      // White O-O-O
      board.square[A1] = WHITE_ROOK;
      board.square[D1] = EMPTY;
    } else if (to == G8) {
      // Black O-O
      board.square[H8] = BLACK_ROOK;
      board.square[F8] = EMPTY;
    } else if (to == C8) {
      // Black O-O-O
      board.square[A8] = BLACK_ROOK;
      board.square[D8] = EMPTY;
    }
  } else {
    // For normal moves, promotions, and regular captures
    board.square[to] =
        captured; // This handles both captures and quiet moves (captured=EMPTY)
  }

  // Rebuild all bitboards and material counts from the restored square array
  _rebuildBitboards();
  _rebuildMaterial();
}

// Helper to rebuild all bitboards from the board.square array
void _rebuildBitboards() {
  board.whiteKing = 0;
  board.whiteQueens = 0;
  board.whiteRooks = 0;
  board.whiteBishops = 0;
  board.whiteKnights = 0;
  board.whitePawns = 0;
  board.blackKing = 0;
  board.blackQueens = 0;
  board.blackRooks = 0;
  board.blackBishops = 0;
  board.blackKnights = 0;
  board.blackPawns = 0;

  for (int i = 0; i < 64; i++) {
    switch (board.square[i]) {
      case WHITE_PAWN:
        board.whitePawns |= BITSET[i];
        break;
      case WHITE_KNIGHT:
        board.whiteKnights |= BITSET[i];
        break;
      case WHITE_BISHOP:
        board.whiteBishops |= BITSET[i];
        break;
      case WHITE_ROOK:
        board.whiteRooks |= BITSET[i];
        break;
      case WHITE_QUEEN:
        board.whiteQueens |= BITSET[i];
        break;
      case WHITE_KING:
        board.whiteKing |= BITSET[i];
        break;
      case BLACK_PAWN:
        board.blackPawns |= BITSET[i];
        break;
      case BLACK_KNIGHT:
        board.blackKnights |= BITSET[i];
        break;
      case BLACK_BISHOP:
        board.blackBishops |= BITSET[i];
        break;
      case BLACK_ROOK:
        board.blackRooks |= BITSET[i];
        break;
      case BLACK_QUEEN:
        board.blackQueens |= BITSET[i];
        break;
      case BLACK_KING:
        board.blackKing |= BITSET[i];
        break;
    }
  }

  board.whitePieces =
      board.whitePawns |
      board.whiteKnights |
      board.whiteBishops |
      board.whiteRooks |
      board.whiteQueens |
      board.whiteKing;
  board.blackPieces =
      board.blackPawns |
      board.blackKnights |
      board.blackBishops |
      board.blackRooks |
      board.blackQueens |
      board.blackKing;
  board.occupiedSquares = board.whitePieces | board.blackPieces;
}

// Helper to update material counts when a piece is captured
void _updateMaterialAndBitboardsForCapture(int captured, int sq) {
  board.Material -= PIECEVALUES[captured];
  switch (captured) {
    case WHITE_PAWN:
      board.totalWhitePawns -= PAWN_VALUE;
      break;
    case WHITE_KNIGHT:
      board.totalWhitePieces -= KNIGHT_VALUE;
      break;
    case WHITE_BISHOP:
      board.totalWhitePieces -= BISHOP_VALUE;
      break;
    case WHITE_ROOK:
      board.totalWhitePieces -= ROOK_VALUE;
      break;
    case WHITE_QUEEN:
      board.totalWhitePieces -= QUEEN_VALUE;
      break;
    case BLACK_PAWN:
      board.totalBlackPawns -= PAWN_VALUE;
      break;
    case BLACK_KNIGHT:
      board.totalBlackPieces -= KNIGHT_VALUE;
      break;
    case BLACK_BISHOP:
      board.totalBlackPieces -= BISHOP_VALUE;
      break;
    case BLACK_ROOK:
      board.totalBlackPieces -= ROOK_VALUE;
      break;
    case BLACK_QUEEN:
      board.totalBlackPieces -= QUEEN_VALUE;
      break;
  }
}

// Helper to update material counts for promotion
void _updateMaterialForPromotion(int pawn, int promotedPiece) {
  board.Material -= PIECEVALUES[pawn];
  board.Material += PIECEVALUES[promotedPiece];
  if (pawn == WHITE_PAWN) {
    board.totalWhitePawns -= PAWN_VALUE;
    board.totalWhitePieces += PIECEVALUES[promotedPiece];
  } else {
    board.totalBlackPawns -= PAWN_VALUE;
    board.totalBlackPieces += PIECEVALUES[promotedPiece];
  }
}

// Helper to rebuild material from scratch
void _rebuildMaterial() {
  board.totalWhitePawns = bitCnt(board.whitePawns) * PAWN_VALUE;
  board.totalBlackPawns = bitCnt(board.blackPawns) * PAWN_VALUE;
  board.totalWhitePieces =
      bitCnt(board.whiteKnights) * KNIGHT_VALUE +
      bitCnt(board.whiteBishops) * BISHOP_VALUE +
      bitCnt(board.whiteRooks) * ROOK_VALUE +
      bitCnt(board.whiteQueens) * QUEEN_VALUE;
  board.totalBlackPieces =
      bitCnt(board.blackKnights) * KNIGHT_VALUE +
      bitCnt(board.blackBishops) * BISHOP_VALUE +
      bitCnt(board.blackRooks) * ROOK_VALUE +
      bitCnt(board.blackQueens) * QUEEN_VALUE;
  board.Material =
      (board.totalWhitePawns + board.totalWhitePieces) -
      (board.totalBlackPawns + board.totalBlackPieces);
}
