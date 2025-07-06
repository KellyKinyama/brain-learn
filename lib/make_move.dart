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

// Helper for debugging, similar to C++ KENNY_DEBUG_MOVES
// void debugMoves(String caller, Move move) {
//   print("DEBUG_MOVES: $caller - Move: ${move.toString()}");
//   board.display();
// }

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
  board.gameLine[board.endOfGame].key = board.hashkey; // Store current hash key

  board.endOfGame++;
  board.endOfSearch++;

  // Update fifty-move rule counter
  board.fiftyMove++;
  if (move.isPawnmove() || move.isCapture()) {
    board.fiftyMove = 0; // Reset on pawn move or capture
  }

  // Update en passant square hash key
  if (board.epSquare != 0) {
    board.hashkey ^= KEY.ep[board.epSquare];
  }
  board.epSquare = 0; // Reset en passant square

  // Update castling rights hash keys
  if ((board.castleWhite & CANCASTLEOO) != 0) board.hashkey ^= KEY.wk;
  if ((board.castleWhite & CANCASTLEOOO) != 0) board.hashkey ^= KEY.wq;
  if ((board.castleBlack & CANCASTLEOO) != 0) board.hashkey ^= KEY.bk;
  if ((board.castleBlack & CANCASTLEOOO) != 0) board.hashkey ^= KEY.bq;

  // Remove old castling rights (if the king or rook moves)
  // This logic is based on the squares involved in castling rights.
  // If a piece moves from A1, H1, E1, A8, H8, E8, castling rights might change.
  if (from == A1 || to == A1) board.castleWhite &= ~CANCASTLEOOO;
  if (from == H1 || to == H1) board.castleWhite &= ~CANCASTLEOO;
  if (from == E1)
    board.castleWhite = 0; // King moves, lose all white castling rights

  if (from == A8 || to == A8) board.castleBlack &= ~CANCASTLEOOO;
  if (from == H8 || to == H8) board.castleBlack &= ~CANCASTLEOO;
  if (from == E8)
    board.castleBlack = 0; // King moves, lose all black castling rights

  // Add new castling rights hash keys (after updates)
  if ((board.castleWhite & CANCASTLEOO) != 0) board.hashkey ^= KEY.wk;
  if ((board.castleWhite & CANCASTLEOOO) != 0) board.hashkey ^= KEY.wq;
  if ((board.castleBlack & CANCASTLEOO) != 0) board.hashkey ^= KEY.bk;
  if ((board.castleBlack & CANCASTLEOOO) != 0) board.hashkey ^= KEY.bq;

  // Update hash key for side to move
  board.hashkey ^= KEY.side;

  // Update bitboards and square array
  // Remove piece from 'from' square
  board.hashkey ^= KEY.keys[from][piece]; // XOR out old piece hash
  board.square[from] = EMPTY;

  // Place piece on 'to' square
  board.hashkey ^= KEY.keys[to][piece]; // XOR in new piece hash
  board.square[to] = piece;

  // Handle captures
  if (captured != EMPTY) {
    board.hashkey ^= KEY.keys[to][captured]; // XOR out captured piece hash
    board.Material -= PIECEVALUES[captured]; // Update material
    if (captured == BLACK_PAWN) board.totalBlackPawns -= PAWN_VALUE;
    if (captured >= BLACK_KNIGHT && captured <= BLACK_QUEEN)
      board.totalBlackPieces -= PIECEVALUES[captured];
    if (captured == WHITE_PAWN) board.totalWhitePawns -= PAWN_VALUE;
    if (captured >= WHITE_KNIGHT && captured <= WHITE_QUEEN)
      board.totalWhitePieces -= PIECEVALUES[captured];
  }

  // Handle specific move types
  if (move.isPawnDoublemove()) {
    // Set en passant square
    board.epSquare = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.hashkey ^= KEY.ep[board.epSquare]; // XOR in new ep hash
  } else if (move.isEnpassant()) {
    // Remove captured pawn from its actual square (not 'to' square)
    int capturedPawnSq = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.square[capturedPawnSq] = EMPTY;
    board.hashkey ^=
        KEY.keys[capturedPawnSq][captured]; // XOR out captured pawn hash
    board.Material -= PIECEVALUES[captured];
    if (captured == BLACK_PAWN) board.totalBlackPawns -= PAWN_VALUE;
    if (captured == WHITE_PAWN) board.totalWhitePawns -= PAWN_VALUE;
  } else if (move.isCastleOO()) {
    // Move rook for O-O
    int rookFrom = (piece == WHITE_KING) ? H1 : H8;
    int rookTo = (piece == WHITE_KING) ? F1 : F8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;

    board.hashkey ^= KEY.keys[rookFrom][rookPiece]; // XOR out old rook hash
    board.square[rookFrom] = EMPTY;
    board.hashkey ^= KEY.keys[rookTo][rookPiece]; // XOR in new rook hash
    board.square[rookTo] = rookPiece;
  } else if (move.isCastleOOO()) {
    // Move rook for O-O-O
    int rookFrom = (piece == WHITE_KING) ? A1 : A8;
    int rookTo = (piece == WHITE_KING) ? D1 : D8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;

    board.hashkey ^= KEY.keys[rookFrom][rookPiece]; // XOR out old rook hash
    board.square[rookFrom] = EMPTY;
    board.hashkey ^= KEY.keys[rookTo][rookPiece]; // XOR in new rook hash
    board.square[rookTo] = rookPiece;
  } else if (move.isPromotion()) {
    // Handle promotion: remove pawn, add promoted piece
    board.hashkey ^=
        KEY.keys[to][piece]; // XOR out pawn (already moved to 'to')
    board.Material -= PIECEVALUES[piece]; // Remove pawn material
    if (piece == WHITE_PAWN) board.totalWhitePawns -= PAWN_VALUE;
    if (piece == BLACK_PAWN) board.totalBlackPawns -= PAWN_VALUE;

    board.square[to] = promotion;
    board.hashkey ^= KEY.keys[to][promotion]; // XOR in promoted piece hash
    board.Material += PIECEVALUES[promotion]; // Add promoted piece material
    if (promotion >= WHITE_KNIGHT && promotion <= WHITE_QUEEN)
      board.totalWhitePieces += PIECEVALUES[promotion];
    if (promotion >= BLACK_KNIGHT && promotion <= BLACK_QUEEN)
      board.totalBlackPieces += PIECEVALUES[promotion];
  }

  // Update bitboards based on square array
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

  // Toggle side to move
  board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;

  // Debugging checks (from C++ KENNY_DEBUG_MOVES)
  // These checks compare the calculated material and piece counts with actual bitboard counts.
  // They are useful for verifying the correctness of makeMove/unmakeMove.
  // For production, these would typically be removed or conditional.
  // if (bitCnt(board.whitePawns) * PAWN_VALUE +
  //         bitCnt(board.whiteKnights) * KNIGHT_VALUE +
  //         bitCnt(board.whiteBishops) * BISHOP_VALUE +
  //         bitCnt(board.whiteRooks) * ROOK_VALUE +
  //         bitCnt(board.whiteQueens) * QUEEN_VALUE !=
  //     board.totalWhitePawns + board.totalWhitePieces) {
  //   print("Inconsistency in white material after makeMove!");
  //   // You might want to display board and move for debugging here
  // }
  // if (bitCnt(board.blackPawns) * PAWN_VALUE +
  //         bitCnt(board.blackKnights) * KNIGHT_VALUE +
  //         bitCnt(board.blackBishops) * BISHOP_VALUE +
  //         bitCnt(board.blackRooks) * ROOK_VALUE +
  //         bitCnt(board.blackQueens) * QUEEN_VALUE !=
  //     board.totalBlackPawns + board.totalBlackPieces) {
  //   print("Inconsistency in black material after makeMove!");
  // }
  // if (board.Material !=
  //     (board.totalWhitePawns +
  //         board.totalWhitePieces +
  //         KING_VALUE - // King value is always there
  //         (board.totalBlackPawns + board.totalBlackPieces + KING_VALUE))) {
  //   print("Inconsistency in total material after makeMove!");
  // }
}

/// Undoes the last move, restoring the board to its previous state.
/// Translates `unmakeMove()` from kennyMakeMove.cpp.
void unmakeMove(Move move) {
  board.endOfGame--;
  board.endOfSearch--;

  // Restore state from gameLine
  GameLineRecord prevRecord = board.gameLine[board.endOfGame];

  int from = move.getFrom();
  int to = move.getTosq();
  int piece = move.getPiec();
  int captured = move.getCapt();
  int promotion = move.getProm();

  // Toggle side to move back
  board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;

  // Restore hash key for side to move (it was toggled in makeMove)
  board.hashkey ^= KEY.side;

  // Restore old castling rights hash keys (from previous record)
  // XOR out current castling keys
  if ((board.castleWhite & CANCASTLEOO) != 0) board.hashkey ^= KEY.wk;
  if ((board.castleWhite & CANCASTLEOOO) != 0) board.hashkey ^= KEY.wq;
  if ((board.castleBlack & CANCASTLEOO) != 0) board.hashkey ^= KEY.bk;
  if ((board.castleBlack & CANCASTLEOOO) != 0) board.hashkey ^= KEY.bq;

  // Restore actual castling rights
  board.castleWhite = prevRecord.castleWhite;
  board.castleBlack = prevRecord.castleBlack;

  // XOR in restored castling keys
  if ((board.castleWhite & CANCASTLEOO) != 0) board.hashkey ^= KEY.wk;
  if ((board.castleWhite & CANCASTLEOOO) != 0) board.hashkey ^= KEY.wq;
  if ((board.castleBlack & CANCASTLEOO) != 0) board.hashkey ^= KEY.bk;
  if ((board.castleBlack & CANCASTLEOOO) != 0) board.hashkey ^= KEY.bq;

  // Restore en passant square hash key
  if (board.epSquare != 0) {
    board.hashkey ^= KEY.ep[board.epSquare];
  }
  board.epSquare = prevRecord.epSquare;
  if (board.epSquare != 0) {
    board.hashkey ^= KEY.ep[board.epSquare];
  }

  // Restore fifty-move rule counter
  board.fiftyMove = prevRecord.fiftyMove;

  // Update bitboards and square array
  // Move piece back from 'to' square to 'from' square
  board.hashkey ^= KEY.keys[to][piece]; // XOR out piece from 'to'
  board.square[to] = EMPTY;
  board.hashkey ^= KEY.keys[from][piece]; // XOR in piece at 'from'
  board.square[from] = piece;

  // Restore captured piece
  if (captured != EMPTY) {
    board.hashkey ^= KEY.keys[to][captured]; // XOR in captured piece at 'to'
    board.square[to] = captured;
    board.Material += PIECEVALUES[captured]; // Restore material
    if (captured == BLACK_PAWN) board.totalBlackPawns += PAWN_VALUE;
    if (captured >= BLACK_KNIGHT && captured <= BLACK_QUEEN)
      board.totalBlackPieces += PIECEVALUES[captured];
    if (captured == WHITE_PAWN) board.totalWhitePawns += PAWN_VALUE;
    if (captured >= WHITE_KNIGHT && captured <= WHITE_QUEEN)
      board.totalWhitePieces += PIECEVALUES[captured];
  }

  // Handle specific move types for unmaking
  if (move.isEnpassant()) {
    // Restore captured pawn at its actual square
    int capturedPawnSq = (piece == WHITE_PAWN) ? (to - 8) : (to + 8);
    board.square[capturedPawnSq] = captured;
    board.hashkey ^=
        KEY.keys[capturedPawnSq][captured]; // XOR in captured pawn hash
    board.Material += PIECEVALUES[captured];
    if (captured == BLACK_PAWN) board.totalBlackPawns += PAWN_VALUE;
    if (captured == WHITE_PAWN) board.totalWhitePawns += PAWN_VALUE;
  } else if (move.isCastleOO()) {
    // Unmove rook for O-O
    int rookFrom = (piece == WHITE_KING) ? H1 : H8;
    int rookTo = (piece == WHITE_KING) ? F1 : F8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;

    board.hashkey ^= KEY.keys[rookTo][rookPiece]; // XOR out rook from 'to'
    board.square[rookTo] = EMPTY;
    board.hashkey ^= KEY.keys[rookFrom][rookPiece]; // XOR in rook at 'from'
    board.square[rookFrom] = rookPiece;
  } else if (move.isCastleOOO()) {
    // Unmove rook for O-O-O
    int rookFrom = (piece == WHITE_KING) ? A1 : A8;
    int rookTo = (piece == WHITE_KING) ? D1 : D8;
    int rookPiece = (piece == WHITE_KING) ? WHITE_ROOK : BLACK_ROOK;

    board.hashkey ^= KEY.keys[rookTo][rookPiece]; // XOR out rook from 'to'
    board.square[rookTo] = EMPTY;
    board.hashkey ^= KEY.keys[rookFrom][rookPiece]; // XOR in rook at 'from'
    board.square[rookFrom] = rookPiece;
  } else if (move.isPromotion()) {
    // Unmake promotion: remove promoted piece, restore pawn
    board.hashkey ^= KEY.keys[to][promotion]; // XOR out promoted piece
    board.Material -= PIECEVALUES[promotion]; // Remove promoted piece material
    if (promotion >= WHITE_KNIGHT && promotion <= WHITE_QUEEN)
      board.totalWhitePieces -= PIECEVALUES[promotion];
    if (promotion >= BLACK_KNIGHT && promotion <= BLACK_QUEEN)
      board.totalBlackPieces -= PIECEVALUES[promotion];

    board.square[to] = piece; // Restore pawn at 'to' square
    board.hashkey ^= KEY.keys[to][piece]; // XOR in pawn hash
    board.Material += PIECEVALUES[piece]; // Add pawn material
    if (piece == WHITE_PAWN) board.totalWhitePawns += PAWN_VALUE;
    if (piece == BLACK_PAWN) board.totalBlackPawns += PAWN_VALUE;
  }

  // Rebuild bitboards from square array (for consistency)
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

  // Debugging checks (from C++ KENNY_DEBUG_MOVES)
  // Same checks as in makeMove to ensure consistency after unmaking.
  // if (bitCnt(board.whitePawns) * PAWN_VALUE +
  //         bitCnt(board.whiteKnights) * KNIGHT_VALUE +
  //         bitCnt(board.whiteBishops) * BISHOP_VALUE +
  //         bitCnt(board.whiteRooks) * ROOK_VALUE +
  //         bitCnt(board.whiteQueens) * QUEEN_VALUE !=
  //     board.totalWhitePawns + board.totalWhitePieces) {
  //   print("Inconsistency in white material after unmakeMove!");
  // }
  // if (bitCnt(board.blackPawns) * PAWN_VALUE +
  //         bitCnt(board.blackKnights) * KNIGHT_VALUE +
  //         bitCnt(board.blackBishops) * BISHOP_VALUE +
  //         bitCnt(board.blackRooks) * ROOK_VALUE +
  //         bitCnt(board.blackQueens) * QUEEN_VALUE !=
  //     board.totalBlackPawns + board.totalBlackPieces) {
  //   print("Inconsistency in black material after unmakeMove!");
  // }
  // if (board.Material !=
  //     (board.totalWhitePawns +
  //         board.totalWhitePieces +
  //         KING_VALUE -
  //         (board.totalBlackPawns + board.totalBlackPieces + KING_VALUE))) {
  //   print("Inconsistency in total material after unmakeMove!");
  // }
}
