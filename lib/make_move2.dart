/// kenny_make_move.dart
///
/// This file contains the implementation for `makeMove` and `unmakeMove` functions.
/// These functions are crucial for updating the board state after a move,
/// including piece positions, material, castling rights, en passant square,
/// fifty-move rule, and Zobrist hash key.
/// This is a corrected version to fix material inconsistency bugs.

import 'defs.dart';
import 'board.dart';
import 'game_line.dart';
import 'move.dart';
import 'hash.dart';
import 'bit_ops.dart'; // For bitCnt

/// Applies a given move to the board, updating all relevant state variables.
/// Translates `makeMove()` from kennyMakeMove.cpp with corrections.
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
  board.epSquare =
      0; // Reset en passant square, will be set later if it's a double pawn push

  // Update castling rights and hash keys
  if ((board.castleWhite & CANCASTLEOO) != 0) board.hashkey ^= KEY.wk;
  if ((board.castleWhite & CANCASTLEOOO) != 0) board.hashkey ^= KEY.wq;
  if ((board.castleBlack & CANCASTLEOO) != 0) board.hashkey ^= KEY.bk;
  if ((board.castleBlack & CANCASTLEOOO) != 0) board.hashkey ^= KEY.bq;

  if (from == A1 || to == A1) board.castleWhite &= ~CANCASTLEOOO;
  if (from == H1 || to == H1) board.castleWhite &= ~CANCASTLEOO;
  if (from == E1) board.castleWhite = 0;

  if (from == A8 || to == A8) board.castleBlack &= ~CANCASTLEOOO;
  if (from == H8 || to == H8) board.castleBlack &= ~CANCASTLEOO;
  if (from == E8) board.castleBlack = 0;

  // Also handle captures on rook squares
  if (to == A1) board.castleWhite &= ~CANCASTLEOOO;
  if (to == H1) board.castleWhite &= ~CANCASTLEOO;
  if (to == A8) board.castleBlack &= ~CANCASTLEOOO;
  if (to == H8) board.castleBlack &= ~CANCASTLEOO;

  if ((board.castleWhite & CANCASTLEOO) != 0) board.hashkey ^= KEY.wk;
  if ((board.castleWhite & CANCASTLEOOO) != 0) board.hashkey ^= KEY.wq;
  if ((board.castleBlack & CANCASTLEOO) != 0) board.hashkey ^= KEY.bk;
  if ((board.castleBlack & CANCASTLEOOO) != 0) board.hashkey ^= KEY.bq;

  board.hashkey ^= KEY.side;

  // --- Core Move Execution ---

  // 1. Remove piece from 'from' square
  board.hashkey ^= KEY.keys[from][piece];
  board.square[from] = EMPTY;

  // 2. Handle captures
  if (captured != EMPTY) {
    if (move.isEnpassant()) {
      int capturedPawnSq = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
      board.square[capturedPawnSq] = EMPTY;
      board.hashkey ^= KEY.keys[capturedPawnSq][captured];
      board.Material -= PIECEVALUES[captured];
      if (captured == WHITE_PAWN) {
        board.totalWhitePawns -= PAWN_VALUE;
      } else {
        board.totalBlackPawns -= PAWN_VALUE;
      }
    } else {
      // Regular capture
      board.hashkey ^= KEY.keys[to][captured];
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
  }

  // 3. Handle piece placement and promotions
  if (move.isPromotion()) {
    // Remove pawn material
    board.Material -= PIECEVALUES[piece];
    if (piece == WHITE_PAWN) {
      board.totalWhitePawns -= PAWN_VALUE;
    } else {
      board.totalBlackPawns -= PAWN_VALUE;
    }

    // Place promoted piece and add its material
    board.square[to] = promotion;
    board.hashkey ^= KEY.keys[to][promotion];
    board.Material += PIECEVALUES[promotion];
    switch (promotion) {
      case WHITE_KNIGHT:
        board.totalWhitePieces += KNIGHT_VALUE;
        break;
      case WHITE_BISHOP:
        board.totalWhitePieces += BISHOP_VALUE;
        break;
      case WHITE_ROOK:
        board.totalWhitePieces += ROOK_VALUE;
        break;
      case WHITE_QUEEN:
        board.totalWhitePieces += QUEEN_VALUE;
        break;
      case BLACK_KNIGHT:
        board.totalBlackPieces += KNIGHT_VALUE;
        break;
      case BLACK_BISHOP:
        board.totalBlackPieces += BISHOP_VALUE;
        break;
      case BLACK_ROOK:
        board.totalBlackPieces += ROOK_VALUE;
        break;
      case BLACK_QUEEN:
        board.totalBlackPieces += QUEEN_VALUE;
        break;
    }
  } else {
    // Not a promotion, just place the moving piece
    board.square[to] = piece;
    board.hashkey ^= KEY.keys[to][piece];
  }

  // 4. Handle other special moves
  if (move.isPawnDoublemove()) {
    board.epSquare = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.hashkey ^= KEY.ep[board.epSquare];
  } else if (move.isCastleOO()) {
    int rookFrom = (piece == WHITE_KING) ? H1 : H8;
    int rookTo = (piece == WHITE_KING) ? F1 : F8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;
    board.hashkey ^=
        KEY.keys[rookFrom][rookPiece] ^ KEY.keys[rookTo][rookPiece];
    board.square[rookFrom] = EMPTY;
    board.square[rookTo] = rookPiece;
  } else if (move.isCastleOOO()) {
    int rookFrom = (piece == WHITE_KING) ? A1 : A8;
    int rookTo = (piece == WHITE_KING) ? D1 : D8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;
    board.hashkey ^=
        KEY.keys[rookFrom][rookPiece] ^ KEY.keys[rookTo][rookPiece];
    board.square[rookFrom] = EMPTY;
    board.square[rookTo] = rookPiece;
  }

  // Update bitboards based on the final square array
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

  board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;

  // Final consistency check
  final int totalMaterialFromPieces =
      (board.totalWhitePawns + board.totalWhitePieces) -
      (board.totalBlackPawns + board.totalBlackPieces);
  if (board.Material != totalMaterialFromPieces) {
    print("Inconsistency in total material after makeMove!");
  }
}

/// Undoes the last move, restoring the board to its previous state.
/// Translates `unmakeMove()` from kennyMakeMove.cpp with corrections.
void unmakeMove(Move move) {
  board.endOfGame--;
  board.endOfSearch--;

  GameLineRecord prevRecord = board.gameLine[board.endOfGame];

  int from = move.getFrom();
  int to = move.getTosq();
  int piece = move.getPiec();
  int captured = move.getCapt();
  int promotion = move.getProm();

  board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;

  // Restore state from gameLine
  board.castleWhite = prevRecord.castleWhite;
  board.castleBlack = prevRecord.castleBlack;
  board.fiftyMove = prevRecord.fiftyMove;
  board.epSquare = prevRecord.epSquare;
  board.hashkey = prevRecord.key; // Restore hash key completely

  // --- Core Un-make Execution ---
  // The hash key is restored from the game line, so we only need to update the board state.
  // The bitboards will be rebuilt at the end for simplicity and correctness.

  // 1. Move piece back from 'to' to 'from'
  board.square[from] = piece;

  // 2. Clear the 'to' square. If it was a regular capture, the captured piece will be placed here.
  board.square[to] = EMPTY;

  // 3. Handle un-doing special move effects
  if (move.isPromotion()) {
    // The piece on the 'to' square was the promotion piece.
    // It's already gone since we set square[to] = EMPTY.
    // Now we just need to restore the captured piece if there was one.
    if (captured != EMPTY) {
      board.square[to] = captured;
    }
  } else if (move.isEnpassant()) {
    // Restore the captured pawn to its actual square
    int capturedPawnSq = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.square[capturedPawnSq] = captured;
    board.square[to] = EMPTY; // The 'to' square was empty in an en passant move
  } else if (move.isCastleOO()) {
    int rookFrom = (piece == WHITE_KING) ? H1 : H8;
    int rookTo = (piece == WHITE_KING) ? F1 : F8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;
    board.square[rookFrom] = rookPiece;
    board.square[rookTo] = EMPTY;
  } else if (move.isCastleOOO()) {
    int rookFrom = (piece == WHITE_KING) ? A1 : A8;
    int rookTo = (piece == WHITE_KING) ? D1 : D8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;
    board.square[rookFrom] = rookPiece;
    board.square[rookTo] = EMPTY;
  } else if (captured != EMPTY) {
    // For regular captures, restore the captured piece on the 'to' square
    board.square[to] = captured;
  }

  // Recalculate all material and bitboards from scratch based on the restored square array
  board.Material = 0;
  board.totalWhitePawns = 0;
  board.totalBlackPawns = 0;
  board.totalWhitePieces = 0;
  board.totalBlackPieces = 0;

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
        board.totalWhitePawns += PAWN_VALUE;
        break;
      case WHITE_KNIGHT:
        board.whiteKnights |= BITSET[i];
        board.totalWhitePieces += KNIGHT_VALUE;
        break;
      case WHITE_BISHOP:
        board.whiteBishops |= BITSET[i];
        board.totalWhitePieces += BISHOP_VALUE;
        break;
      case WHITE_ROOK:
        board.whiteRooks |= BITSET[i];
        board.totalWhitePieces += ROOK_VALUE;
        break;
      case WHITE_QUEEN:
        board.whiteQueens |= BITSET[i];
        board.totalWhitePieces += QUEEN_VALUE;
        break;
      case WHITE_KING:
        board.whiteKing |= BITSET[i];
        break;
      case BLACK_PAWN:
        board.blackPawns |= BITSET[i];
        board.totalBlackPawns += PAWN_VALUE;
        break;
      case BLACK_KNIGHT:
        board.blackKnights |= BITSET[i];
        board.totalBlackPieces += KNIGHT_VALUE;
        break;
      case BLACK_BISHOP:
        board.blackBishops |= BITSET[i];
        board.totalBlackPieces += BISHOP_VALUE;
        break;
      case BLACK_ROOK:
        board.blackRooks |= BITSET[i];
        board.totalBlackPieces += ROOK_VALUE;
        break;
      case BLACK_QUEEN:
        board.blackQueens |= BITSET[i];
        board.totalBlackPieces += QUEEN_VALUE;
        break;
      case BLACK_KING:
        board.blackKing |= BITSET[i];
        break;
    }
  }

  board.Material =
      (board.totalWhitePawns + board.totalWhitePieces) -
      (board.totalBlackPawns + board.totalBlackPieces);
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

  // Final consistency check
  final int totalMaterialFromPieces =
      (board.totalWhitePawns + board.totalWhitePieces) -
      (board.totalBlackPawns + board.totalBlackPieces);
  if (board.Material != totalMaterialFromPieces) {
    print("Inconsistency in total material after unmakeMove!");
  }
}
