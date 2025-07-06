/// kenny_move_gen.dart
///
/// This file contains functions for generating pseudo-legal and legal chess moves.
/// It translates `movegen()` and `captgen()` from kennyMoveGen.cpp.
/// This is a critical and complex part of the chess engine.

import 'defs.dart';
import 'board.dart';
import 'move.dart';
import 'bit_ops.dart'; // For firstOne, lastOne, bitCnt
import 'make_move.dart';
import 'utils.dart'; // For makeMove, unmakeMove

/// Generates all legal moves from the current board position.
/// It first generates pseudo-legal moves (moves that are geometrically valid)
/// and then filters out any move that would leave the king in check.
/// Translates `movegen()` from kennyMoveGen.cpp.
/// Returns the new end index for the move buffer.
int movegen(int moveBufStartIdx) {
  int currentMoveIdx = moveBufStartIdx;
  int from, to;
  BitMap targetBitmap;
  BitMap pieceBitmap;

  // Clear the move buffer for the current ply
  board.moveBufLen[board.endOfGame] = moveBufStartIdx;
  board.moveBufLen[board.endOfGame + 1] =
      moveBufStartIdx; // Reset next ply's start

  // Determine the side to move
  if (board.nextMove == WHITE_MOVE) {
    // Generate White's moves
    // =====================================================================
    // Pawns
    // =====================================================================
    pieceBitmap = board.whitePawns;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from]; // Clear this bit

      // Single pawn push
      targetBitmap = WHITE_PAWN_MOVES[from] & ~board.occupiedSquares;
      if (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        if (RANKS[to] == 8) {
          // Promotion
          addPromotionMoves(from, to, WHITE_PAWN, EMPTY, currentMoveIdx);
          currentMoveIdx += 4;
        } else {
          // Normal push
          addMove(from, to, WHITE_PAWN, EMPTY, EMPTY, currentMoveIdx++);

          // Double pawn push
          if (RANKS[from] == 2) {
            targetBitmap =
                WHITE_PAWN_DOUBLE_MOVES[from] & ~board.occupiedSquares;
            if (targetBitmap != 0 &&
                (BITSET[from + 8] & board.occupiedSquares) == 0) {
              to = firstOne(targetBitmap);
              addMove(
                from,
                to,
                WHITE_PAWN,
                EMPTY,
                EMPTY,
                currentMoveIdx++,
                isPawnDouble: true,
              );
            }
          }
        }
      }

      // Pawn captures
      targetBitmap = WHITE_PAWN_ATTACKS[from] & board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];
        int capturedPiece = board.square[to];

        if (RANKS[to] == 8) {
          addPromotionMoves(
            from,
            to,
            WHITE_PAWN,
            capturedPiece,
            currentMoveIdx,
          );
          currentMoveIdx += 4;
        } else {
          addMove(from, to, WHITE_PAWN, capturedPiece, EMPTY, currentMoveIdx++);
        }
      }

      // En passant captures
      if (board.epSquare != 0) {
        if ((WHITE_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
          if (RANKS[from] == 5) {
            addMove(
              from,
              board.epSquare,
              WHITE_PAWN,
              BLACK_PAWN,
              EMPTY,
              currentMoveIdx++,
              isEnpassant: true,
            );
          }
        }
      }
    }

    // =====================================================================
    // Knights, Bishops, Rooks, Queens
    // =====================================================================
    generatePieceMoves(WHITE_KNIGHT, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];
    generatePieceMoves(WHITE_BISHOP, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];
    generatePieceMoves(WHITE_ROOK, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];
    generatePieceMoves(WHITE_QUEEN, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];

    // =====================================================================
    // King
    // =====================================================================
    pieceBitmap = board.whiteKing;
    if (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      targetBitmap = KING_ATTACKS[from] & ~board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];
        addMove(
          from,
          to,
          WHITE_KING,
          board.square[to],
          EMPTY,
          currentMoveIdx++,
        );
      }

      // Castling
      if ((board.castleWhite & CANCASTLEOO) != 0) {
        if ((board.occupiedSquares & (BITSET[F1] | BITSET[G1])) == 0) {
          if (!isAttacked(E1, BLACK_MOVE) &&
              !isAttacked(F1, BLACK_MOVE) &&
              !isAttacked(G1, BLACK_MOVE)) {
            addMove(
              E1,
              G1,
              WHITE_KING,
              EMPTY,
              EMPTY,
              currentMoveIdx++,
              isCastle: true,
            );
          }
        }
      }
      if ((board.castleWhite & CANCASTLEOOO) != 0) {
        if ((board.occupiedSquares & (BITSET[B1] | BITSET[C1] | BITSET[D1])) ==
            0) {
          if (!isAttacked(E1, BLACK_MOVE) &&
              !isAttacked(D1, BLACK_MOVE) &&
              !isAttacked(C1, BLACK_MOVE)) {
            addMove(
              E1,
              C1,
              WHITE_KING,
              EMPTY,
              EMPTY,
              currentMoveIdx++,
              isCastle: true,
            );
          }
        }
      }
    }
  } else {
    // Generate Black's moves (similar logic)
    // =====================================================================
    // Pawns
    // =====================================================================
    pieceBitmap = board.blackPawns;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap = BLACK_PAWN_MOVES[from] & ~board.occupiedSquares;
      if (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        if (RANKS[to] == 1) {
          addPromotionMoves(from, to, BLACK_PAWN, EMPTY, currentMoveIdx);
          currentMoveIdx += 4;
        } else {
          addMove(from, to, BLACK_PAWN, EMPTY, EMPTY, currentMoveIdx++);
          if (RANKS[from] == 7) {
            targetBitmap =
                BLACK_PAWN_DOUBLE_MOVES[from] & ~board.occupiedSquares;
            if (targetBitmap != 0 &&
                (BITSET[from - 8] & board.occupiedSquares) == 0) {
              to = firstOne(targetBitmap);
              addMove(
                from,
                to,
                BLACK_PAWN,
                EMPTY,
                EMPTY,
                currentMoveIdx++,
                isPawnDouble: true,
              );
            }
          }
        }
      }

      targetBitmap = BLACK_PAWN_ATTACKS[from] & board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];
        int capturedPiece = board.square[to];
        if (RANKS[to] == 1) {
          addPromotionMoves(
            from,
            to,
            BLACK_PAWN,
            capturedPiece,
            currentMoveIdx,
          );
          currentMoveIdx += 4;
        } else {
          addMove(from, to, BLACK_PAWN, capturedPiece, EMPTY, currentMoveIdx++);
        }
      }

      if (board.epSquare != 0) {
        if ((BLACK_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
          if (RANKS[from] == 4) {
            addMove(
              from,
              board.epSquare,
              BLACK_PAWN,
              WHITE_PAWN,
              EMPTY,
              currentMoveIdx++,
              isEnpassant: true,
            );
          }
        }
      }
    }

    // =====================================================================
    // Knights, Bishops, Rooks, Queens
    // =====================================================================
    generatePieceMoves(BLACK_KNIGHT, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];
    generatePieceMoves(BLACK_BISHOP, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];
    generatePieceMoves(BLACK_ROOK, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];
    generatePieceMoves(BLACK_QUEEN, currentMoveIdx);
    currentMoveIdx = board.moveBufLen[board.endOfGame + 1];

    // =====================================================================
    // King
    // =====================================================================
    pieceBitmap = board.blackKing;
    if (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      targetBitmap = KING_ATTACKS[from] & ~board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];
        addMove(
          from,
          to,
          BLACK_KING,
          board.square[to],
          EMPTY,
          currentMoveIdx++,
        );
      }

      if ((board.castleBlack & CANCASTLEOO) != 0) {
        if ((board.occupiedSquares & (BITSET[F8] | BITSET[G8])) == 0) {
          if (!isAttacked(E8, WHITE_MOVE) &&
              !isAttacked(F8, WHITE_MOVE) &&
              !isAttacked(G8, WHITE_MOVE)) {
            addMove(
              E8,
              G8,
              BLACK_KING,
              EMPTY,
              EMPTY,
              currentMoveIdx++,
              isCastle: true,
            );
          }
        }
      }
      if ((board.castleBlack & CANCASTLEOOO) != 0) {
        if ((board.occupiedSquares & (BITSET[B8] | BITSET[C8] | BITSET[D8])) ==
            0) {
          if (!isAttacked(E8, WHITE_MOVE) &&
              !isAttacked(D8, WHITE_MOVE) &&
              !isAttacked(C8, WHITE_MOVE)) {
            addMove(
              E8,
              C8,
              BLACK_KING,
              EMPTY,
              EMPTY,
              currentMoveIdx++,
              isCastle: true,
            );
          }
        }
      }
    }
  }

  // Filter out illegal moves (moves that leave own king in check)
  int legalMovesCount = moveBufStartIdx;
  for (int i = moveBufStartIdx; i < currentMoveIdx; i++) {
    Move currentMove = board.moveBuffer[i];
    makeMove(currentMove); // Temporarily make the move
    if (!isOwnKingAttacked()) {
      // If king is not attacked after the move, it's legal
      board.moveBuffer[legalMovesCount] = currentMove; // Keep the legal move
      legalMovesCount++;
    }
    unmakeMove(currentMove); // Unmake the move
  }

  board.moveBufLen[board.endOfGame + 1] = legalMovesCount;
  return legalMovesCount;
}

// Helper to add a standard move
void addMove(
  int from,
  int to,
  int piece,
  int captured,
  int promotion,
  int index, {
  bool isCastle = false,
  bool isEnpassant = false,
  bool isPawnDouble = false,
}) {
  board.moveBuffer[index].clear();
  board.moveBuffer[index].setFrom(from);
  board.moveBuffer[index].setTosq(to);
  board.moveBuffer[index].setPiec(piece);
  board.moveBuffer[index].setCapt(captured);
  board.moveBuffer[index].setProm(promotion);
  board.moveBuffer[index].setCastle(isCastle);
  board.moveBuffer[index].setEnpassant(isEnpassant);
  board.moveBuffer[index].setPawnDoubleMove(isPawnDouble);
  board.moveBufLen[board.endOfGame + 1] = index + 1;
}

// Helper to add all 4 promotion moves
void addPromotionMoves(
  int from,
  int to,
  int piece,
  int captured,
  int startIndex,
) {
  int promPiece = (piece == WHITE_PAWN) ? WHITE_QUEEN : BLACK_QUEEN;
  addMove(from, to, piece, captured, promPiece, startIndex);
  promPiece = (piece == WHITE_PAWN) ? WHITE_ROOK : BLACK_ROOK;
  addMove(from, to, piece, captured, promPiece, startIndex + 1);
  promPiece = (piece == WHITE_PAWN) ? WHITE_BISHOP : BLACK_BISHOP;
  addMove(from, to, piece, captured, promPiece, startIndex + 2);
  promPiece = (piece == WHITE_PAWN) ? WHITE_KNIGHT : BLACK_KNIGHT;
  addMove(from, to, piece, captured, promPiece, startIndex + 3);
}

// Helper to generate moves for non-pawn pieces
void generatePieceMoves(int piece, int startIndex) {
  BitMap pieceBitmap;
  BitMap friendlyPieces;
  int currentMoveIdx = startIndex;

  if (PIECE_IS_WHITE[piece]) {
    pieceBitmap = board.getBitboardForPiece(piece);
    friendlyPieces = board.whitePieces;
  } else {
    pieceBitmap = board.getBitboardForPiece(piece);
    friendlyPieces = board.blackPieces;
  }

  while (pieceBitmap != 0) {
    int from = firstOne(pieceBitmap);
    pieceBitmap ^= BITSET[from];
    BitMap attacks = getAttacksForPiece(piece, from, board.occupiedSquares);
    attacks &= ~friendlyPieces;

    while (attacks != 0) {
      int to = firstOne(attacks);
      attacks ^= BITSET[to];
      addMove(from, to, piece, board.square[to], EMPTY, currentMoveIdx++);
    }
  }
}

BitMap getAttacksForPiece(int piece, int from, BitMap occupied) {
  switch (piece) {
    case WHITE_KNIGHT:
    case BLACK_KNIGHT:
      return KNIGHT_ATTACKS[from];
    case WHITE_BISHOP:
    case BLACK_BISHOP:
      return getBishopAttacks(from, occupied);
    case WHITE_ROOK:
    case BLACK_ROOK:
      return getRookAttacks(from, occupied);
    case WHITE_QUEEN:
    case BLACK_QUEEN:
      return getBishopAttacks(from, occupied) | getRookAttacks(from, occupied);
    case WHITE_KING:
    case BLACK_KING:
      return KING_ATTACKS[from];
    default:
      return 0;
  }
}

/// Helper function to get attacks for a bishop from a given square.
/// Helper function to get attacks for a bishop from a given square, considering blockers.
/// This is a simplified version; a full magic bitboard implementation would be more complex.
/// It traces rays in diagonal directions until a blocker or board edge is hit.
BitMap getBishopAttacks(int square, BitMap occupied) {
  BitMap attacks = 0;
  int r = square ~/ 8;
  int f = square % 8;

  // Northeast (rank++, file++)
  for (int i = 1; r + i < 8 && f + i < 8; i++) {
    int targetSq = square + i * 9;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  // Northwest (rank++, file--)
  for (int i = 1; r + i < 8 && f - i >= 0; i++) {
    int targetSq = square + i * 7;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  // Southeast (rank--, file++)
  for (int i = 1; r - i >= 0 && f + i < 8; i++) {
    int targetSq = square - i * 7;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  // Southwest (rank--, file--)
  for (int i = 1; r - i >= 0 && f - i >= 0; i++) {
    int targetSq = square - i * 9;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  return attacks;
}

/// Helper function to get attacks for a rook from a given square.
BitMap getRookAttacks(int square, BitMap occupied) {
  BitMap attacks = 0;
  int r, f;
  int i;

  r = square ~/ 8;
  f = square % 8;

  // North
  for (i = r + 1; i < 8; i++) {
    int to = i * 8 + f;
    attacks |= BITSET[to];
    if ((occupied & BITSET[to]) != 0) break;
  }
  // South
  for (i = r - 1; i >= 0; i--) {
    int to = i * 8 + f;
    attacks |= BITSET[to];
    if ((occupied & BITSET[to]) != 0) break;
  }
  // East
  for (i = f + 1; i < 8; i++) {
    int to = r * 8 + i;
    attacks |= BITSET[to];
    if ((occupied & BITSET[to]) != 0) break;
  }
  // West
  for (i = f - 1; i >= 0; i--) {
    int to = r * 8 + i;
    attacks |= BITSET[to];
    if ((occupied & BITSET[to]) != 0) break;
  }
  return attacks;
}

/// Checks if the king of the side that just moved is in check.
/// This is called after a move is made, so `board.nextMove` has been flipped.
BOOLTYPE isOwnKingAttacked() {
  // If it's now White's turn, it means Black just moved.
  if (board.nextMove == WHITE_MOVE) {
    // Check if the black king is attacked by white pieces.
    if (board.blackKing == 0) return true; // King captured, illegal state
    return isAttacked(firstOne(board.blackKing), WHITE_MOVE);
  } else {
    // It's now Black's turn, so White just moved.
    // Check if the white king is attacked by black pieces.
    if (board.whiteKing == 0) return true; // King captured, illegal state
    return isAttacked(firstOne(board.whiteKing), BLACK_MOVE);
  }
}

/// Checks if a given square is attacked by any piece of the specified attacking side.
/// [targetSquare] The square to check for attacks.
/// [attackingSide] The side performing the attack (WHITE_MOVE or BLACK_MOVE).
BOOLTYPE isAttacked(int targetSquare, int attackingSide) {
  if (attackingSide == WHITE_MOVE) {
    // Check if targetSquare is attacked BY WHITE pieces
    if ((BLACK_PAWN_ATTACKS[targetSquare] & board.whitePawns) != 0) return true;
    if ((KNIGHT_ATTACKS[targetSquare] & board.whiteKnights) != 0) return true;
    if ((KING_ATTACKS[targetSquare] & board.whiteKing) != 0) return true;
    if ((getRookAttacks(targetSquare, board.occupiedSquares) &
            (board.whiteRooks | board.whiteQueens)) !=
        0)
      return true;
    if ((getBishopAttacks(targetSquare, board.occupiedSquares) &
            (board.whiteBishops | board.whiteQueens)) !=
        0)
      return true;
  } else {
    // Check if targetSquare is attacked BY BLACK pieces
    if ((WHITE_PAWN_ATTACKS[targetSquare] & board.blackPawns) != 0) return true;
    if ((KNIGHT_ATTACKS[targetSquare] & board.blackKnights) != 0) return true;
    if ((KING_ATTACKS[targetSquare] & board.blackKing) != 0) return true;
    if ((getRookAttacks(targetSquare, board.occupiedSquares) &
            (board.blackRooks | board.blackQueens)) !=
        0)
      return true;
    if ((getBishopAttacks(targetSquare, board.occupiedSquares) &
            (board.blackBishops | board.blackQueens)) !=
        0)
      return true;
  }
  return false;
}

/// A lookup table to quickly determine if a piece is white.
/// Indexed by the piece's integer value.
const List<bool> PIECE_IS_WHITE = [
  false, // 0: EMPTY
  true, // 1: WHITE_PAWN
  true, // 2: WHITE_KING
  true, // 3: WHITE_KNIGHT
  false, // 4: (unused)
  true, // 5: WHITE_BISHOP
  true, // 6: WHITE_ROOK
  true, // 7: WHITE_QUEEN
  false, // 8: (unused color bit)
  false, // 9: BLACK_PAWN
  false, // 10: BLACK_KING
  false, // 11: BLACK_KNIGHT
  false, // 12: (unused)
  false, // 13: BLACK_BISHOP
  false, // 14: BLACK_ROOK
  false, // 15: BLACK_QUEEN
];
