/// kenny_move_gen.dart
///
/// This file contains functions for generating pseudo-legal and legal chess moves.
/// It translates `movegen()` and `captgen()` from kennyMoveGen.cpp.
/// This is a critical and complex part of the chess engine.

import 'defs.dart';
import 'board.dart';
import 'move.dart';
import 'bit_ops.dart'; // For firstOne, lastOne, bitCnt
import 'make_move2.dart';
import 'utils.dart'; // For makeMove, unmakeMove

/// Generates all pseudo-legal moves from the current board position.
/// Pseudo-legal moves are moves that are geometrically valid but might
/// leave the king in check. Legality checks are performed later.
/// Translates `movegen()` from kennyMoveGen.cpp.
/// Returns the new length of the move buffer.
int movegen(int moveBufStartIdx) {
  int currentMoveIdx = moveBufStartIdx;
  int from, to;
  BitMap targetBitmap;
  BitMap pieceBitmap;

  // Clear the move buffer for the current ply
  board.moveBufLen[board.endOfGame + 1] = 0; // Reset next ply's start

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
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_QUEEN);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_ROOK);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_BISHOP);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_KNIGHT);
          currentMoveIdx++;
        } else {
          // Normal push
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          currentMoveIdx++;

          // Double pawn push
          if (RANKS[from] == 2) {
            targetBitmap =
                WHITE_PAWN_DOUBLE_MOVES[from] & ~board.occupiedSquares;
            if (targetBitmap != 0 &&
                (BITSET[from + 8] & board.occupiedSquares) == 0) {
              // Check intermediate square is empty
              to = firstOne(targetBitmap);
              board.moveBuffer[currentMoveIdx].clear();
              board.moveBuffer[currentMoveIdx].setFrom(from);
              board.moveBuffer[currentMoveIdx].setTosq(to);
              board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
              board.moveBuffer[currentMoveIdx].setPawnDoubleMove(true);
              currentMoveIdx++;
            }
          }
        }
      }

      // Pawn captures
      targetBitmap = WHITE_PAWN_ATTACKS[from] & board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to]; // Clear this bit

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        if (RANKS[to] == 8) {
          // Promotion with capture
          board.moveBuffer[currentMoveIdx].setProm(WHITE_QUEEN);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_ROOK);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_BISHOP);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_KNIGHT);
          currentMoveIdx++;
        } else {
          currentMoveIdx++;
        }
      }

      // En passant captures
      if (board.epSquare != 0) {
        if (from == board.epSquare + 1 || from == board.epSquare - 1) {
          // Check if pawn is on the 5th rank (for white pawn)
          if (RANKS[from] == 5) {
            // Check if the pawn attacks the en passant square
            if ((WHITE_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
              board.moveBuffer[currentMoveIdx].clear();
              board.moveBuffer[currentMoveIdx].setFrom(from);
              board.moveBuffer[currentMoveIdx].setTosq(board.epSquare);
              board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
              board.moveBuffer[currentMoveIdx].setCapt(
                BLACK_PAWN,
              ); // Captured pawn is always black pawn
              board.moveBuffer[currentMoveIdx].setEnpassant(true);
              currentMoveIdx++;
            }
          }
        }
      }
    }

    // =====================================================================
    // Knights
    // =====================================================================
    pieceBitmap = board.whiteKnights;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap =
          KNIGHT_ATTACKS[from] &
          ~board.whitePieces; // Attacks anything not own piece
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(WHITE_KNIGHT);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // Bishops
    // =====================================================================
    pieceBitmap = board.whiteBishops;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      // Sliders need to calculate attacks based on occupied squares
      // This is where magic bitboards or direct ray tracing comes in.
      // For now, a simplified approach (will need full magic bitboard implementation)
      // This part is a placeholder for the actual sliding piece attack generation.
      // The C++ uses macros like SLIDEA8H1MOVES.
      // We need to simulate the sliding attacks based on current board.occupiedSquares.
      targetBitmap =
          getBishopAttacks(from, board.occupiedSquares) & ~board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(WHITE_BISHOP);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // Rooks
    // =====================================================================
    pieceBitmap = board.whiteRooks;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap =
          getRookAttacks(from, board.occupiedSquares) & ~board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(WHITE_ROOK);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // Queens
    // =====================================================================
    pieceBitmap = board.whiteQueens;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap =
          (getBishopAttacks(from, board.occupiedSquares) |
              getRookAttacks(from, board.occupiedSquares)) &
          ~board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(WHITE_QUEEN);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // King
    // =====================================================================
    pieceBitmap = board.whiteKing; // Should only be one bit set
    if (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);

      targetBitmap = KING_ATTACKS[from] & ~board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(WHITE_KING);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }

      // Castling
      // King-side castling (O-O)
      if ((board.castleWhite & CANCASTLEOO) != 0) {
        // Check if squares F1 and G1 are empty and not attacked
        if (((board.occupiedSquares & (BITSET[F1] | BITSET[G1])) == 0) &&
            !isAttacked(board.blackPieces, E1) && // King's current square
            !isAttacked(board.blackPieces, F1) && // Square it passes through
            !isAttacked(board.blackPieces, G1)) {
          // Square it lands on
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(E1);
          board.moveBuffer[currentMoveIdx].setTosq(G1);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_KING);
          board.moveBuffer[currentMoveIdx].setCastle(true);
          currentMoveIdx++;
        }
      }
      // Queen-side castling (O-O-O)
      if ((board.castleWhite & CANCASTLEOOO) != 0) {
        // Check if squares B1, C1, D1 are empty and not attacked
        if (((board.occupiedSquares & (BITSET[B1] | BITSET[C1] | BITSET[D1])) ==
                0) &&
            !isAttacked(board.blackPieces, E1) && // King's current square
            !isAttacked(board.blackPieces, D1) && // Square it passes through
            !isAttacked(board.blackPieces, C1)) {
          // Square it lands on
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(E1);
          board.moveBuffer[currentMoveIdx].setTosq(C1);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_KING);
          board.moveBuffer[currentMoveIdx].setCastle(true);
          currentMoveIdx++;
        }
      }
    }
  } else {
    // Generate Black's moves (similar logic, mirrored)
    // =====================================================================
    // Pawns
    // =====================================================================
    pieceBitmap = board.blackPawns;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      // Single pawn push
      targetBitmap = BLACK_PAWN_MOVES[from] & ~board.occupiedSquares;
      if (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        if (RANKS[to] == 1) {
          // Promotion
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_QUEEN);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_ROOK);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_BISHOP);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_KNIGHT);
          currentMoveIdx++;
        } else {
          // Normal push
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          currentMoveIdx++;

          // Double pawn push
          if (RANKS[from] == 7) {
            targetBitmap =
                BLACK_PAWN_DOUBLE_MOVES[from] & ~board.occupiedSquares;
            if (targetBitmap != 0 &&
                (BITSET[from - 8] & board.occupiedSquares) == 0) {
              // Check intermediate square is empty
              to = firstOne(targetBitmap);
              board.moveBuffer[currentMoveIdx].clear();
              board.moveBuffer[currentMoveIdx].setFrom(from);
              board.moveBuffer[currentMoveIdx].setTosq(to);
              board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
              board.moveBuffer[currentMoveIdx].setPawnDoubleMove(true);
              currentMoveIdx++;
            }
          }
        }
      }

      // Pawn captures
      targetBitmap = BLACK_PAWN_ATTACKS[from] & board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        if (RANKS[to] == 1) {
          // Promotion with capture
          board.moveBuffer[currentMoveIdx].setProm(BLACK_QUEEN);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_ROOK);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_BISHOP);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_KNIGHT);
          currentMoveIdx++;
        } else {
          currentMoveIdx++;
        }
      }

      // En passant captures
      if (board.epSquare != 0) {
        if (from == board.epSquare + 1 || from == board.epSquare - 1) {
          // Check if pawn is on the 4th rank (for black pawn)
          if (RANKS[from] == 4) {
            // Check if the pawn attacks the en passant square
            if ((BLACK_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
              board.moveBuffer[currentMoveIdx].clear();
              board.moveBuffer[currentMoveIdx].setFrom(from);
              board.moveBuffer[currentMoveIdx].setTosq(board.epSquare);
              board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
              board.moveBuffer[currentMoveIdx].setCapt(
                WHITE_PAWN,
              ); // Captured pawn is always white pawn
              board.moveBuffer[currentMoveIdx].setEnpassant(true);
              currentMoveIdx++;
            }
          }
        }
      }
    }

    // =====================================================================
    // Knights
    // =====================================================================
    pieceBitmap = board.blackKnights;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap = KNIGHT_ATTACKS[from] & ~board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(BLACK_KNIGHT);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // Bishops
    // =====================================================================
    pieceBitmap = board.blackBishops;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap =
          getBishopAttacks(from, board.occupiedSquares) & ~board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(BLACK_BISHOP);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // Rooks
    // =====================================================================
    pieceBitmap = board.blackRooks;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap =
          getRookAttacks(from, board.occupiedSquares) & ~board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(BLACK_ROOK);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // Queens
    // =====================================================================
    pieceBitmap = board.blackQueens;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      targetBitmap =
          (getBishopAttacks(from, board.occupiedSquares) |
              getRookAttacks(from, board.occupiedSquares)) &
          ~board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(BLACK_QUEEN);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }
    }

    // =====================================================================
    // King
    // =====================================================================
    pieceBitmap = board.blackKing; // Should only be one bit set
    if (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);

      targetBitmap = KING_ATTACKS[from] & ~board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(BLACK_KING);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        currentMoveIdx++;
      }

      // Castling
      // King-side castling (O-O)
      if ((board.castleBlack & CANCASTLEOO) != 0) {
        // Check if squares F8 and G8 are empty and not attacked
        if (((board.occupiedSquares & (BITSET[F8] | BITSET[G8])) == 0) &&
            !isAttacked(board.whitePieces, E8) &&
            !isAttacked(board.whitePieces, F8) &&
            !isAttacked(board.whitePieces, G8)) {
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(E8);
          board.moveBuffer[currentMoveIdx].setTosq(G8);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_KING);
          board.moveBuffer[currentMoveIdx].setCastle(true);
          currentMoveIdx++;
        }
      }
      // Queen-side castling (O-O-O)
      if ((board.castleBlack & CANCASTLEOOO) != 0) {
        // Check if squares B8, C8, D8 are empty and not attacked
        if (((board.occupiedSquares & (BITSET[B8] | BITSET[C8] | BITSET[D8])) ==
                0) &&
            !isAttacked(board.whitePieces, E8) &&
            !isAttacked(board.whitePieces, D8) &&
            !isAttacked(board.whitePieces, C8)) {
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(E8);
          board.moveBuffer[currentMoveIdx].setTosq(C8);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_KING);
          board.moveBuffer[currentMoveIdx].setCastle(true);
          currentMoveIdx++;
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

  board.moveBufLen[board.endOfGame + 1] =
      legalMovesCount; // Update next ply's start
  return legalMovesCount;
}

/// Generates only pseudo-legal capture and promotion moves.
/// Translates `captgen()` from kennyMoveGen.cpp.
/// Returns the new length of the move buffer, with captures sorted by SEE.
int captgen(int moveBufStartIdx) {
  int currentMoveIdx = moveBufStartIdx;
  int from, to;
  BitMap targetBitmap;
  BitMap pieceBitmap;

  // Clear the move buffer for the current ply
  board.moveBufLen[board.endOfGame + 1] = 0; // Reset next ply's start

  // Determine the side to move
  if (board.nextMove == WHITE_MOVE) {
    // Generate White's captures and promotions
    // =====================================================================
    // Pawns
    // =====================================================================
    pieceBitmap = board.whitePawns;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      // Pawn captures
      targetBitmap = WHITE_PAWN_ATTACKS[from] & board.blackPieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        if (RANKS[to] == 8) {
          // Promotion with capture
          board.moveBuffer[currentMoveIdx].setProm(WHITE_QUEEN);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_ROOK);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_BISHOP);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_KNIGHT);
          currentMoveIdx++;
        } else {
          currentMoveIdx++;
        }
      }

      // En passant captures
      if (board.epSquare != 0) {
        if (from == board.epSquare + 1 || from == board.epSquare - 1) {
          if (RANKS[from] == 5) {
            if ((WHITE_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
              board.moveBuffer[currentMoveIdx].clear();
              board.moveBuffer[currentMoveIdx].setFrom(from);
              board.moveBuffer[currentMoveIdx].setTosq(board.epSquare);
              board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
              board.moveBuffer[currentMoveIdx].setCapt(BLACK_PAWN);
              board.moveBuffer[currentMoveIdx].setEnpassant(true);
              currentMoveIdx++;
            }
          }
        }
      }

      // Pawn promotions (non-capturing)
      targetBitmap = WHITE_PAWN_MOVES[from] & ~board.occupiedSquares;
      if (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        if (RANKS[to] == 8) {
          // Promotion
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_QUEEN);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_ROOK);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_BISHOP);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(WHITE_KNIGHT);
          currentMoveIdx++;
        }
      }
    }

    // =====================================================================
    // Knights, Bishops, Rooks, Queens, King (captures only)
    // =====================================================================
    List<int> whitePieces = [
      WHITE_KNIGHT,
      WHITE_BISHOP,
      WHITE_ROOK,
      WHITE_QUEEN,
      WHITE_KING,
    ];
    List<BitMap> whiteBitboards = [
      board.whiteKnights,
      board.whiteBishops,
      board.whiteRooks,
      board.whiteQueens,
      board.whiteKing,
    ];

    for (int i = 0; i < whitePieces.length; i++) {
      int pieceType = whitePieces[i];
      pieceBitmap = whiteBitboards[i];

      while (pieceBitmap != 0) {
        from = firstOne(pieceBitmap);
        pieceBitmap ^= BITSET[from];

        BitMap attacks;
        if (pieceType == WHITE_KNIGHT) {
          attacks = KNIGHT_ATTACKS[from];
        } else if (pieceType == WHITE_KING) {
          attacks = KING_ATTACKS[from];
        } else if (pieceType == WHITE_BISHOP) {
          attacks = getBishopAttacks(from, board.occupiedSquares);
        } else if (pieceType == WHITE_ROOK) {
          attacks = getRookAttacks(from, board.occupiedSquares);
        } else {
          // QUEEN
          attacks =
              getBishopAttacks(from, board.occupiedSquares) |
              getRookAttacks(from, board.occupiedSquares);
        }

        targetBitmap = attacks & board.blackPieces; // Only captures
        while (targetBitmap != 0) {
          to = firstOne(targetBitmap);
          targetBitmap ^= BITSET[to];

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(pieceType);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          currentMoveIdx++;
        }
      }
    }
  } else {
    // Generate Black's captures and promotions (similar logic, mirrored)
    // =====================================================================
    // Pawns
    // =====================================================================
    pieceBitmap = board.blackPawns;
    while (pieceBitmap != 0) {
      from = firstOne(pieceBitmap);
      pieceBitmap ^= BITSET[from];

      // Pawn captures
      targetBitmap = BLACK_PAWN_ATTACKS[from] & board.whitePieces;
      while (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        targetBitmap ^= BITSET[to];

        board.moveBuffer[currentMoveIdx].clear();
        board.moveBuffer[currentMoveIdx].setFrom(from);
        board.moveBuffer[currentMoveIdx].setTosq(to);
        board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
        board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
        if (RANKS[to] == 1) {
          // Promotion with capture
          board.moveBuffer[currentMoveIdx].setProm(BLACK_QUEEN);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_ROOK);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_BISHOP);
          currentMoveIdx++;
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_KNIGHT);
          currentMoveIdx++;
        } else {
          currentMoveIdx++;
        }
      }

      // En passant captures
      if (board.epSquare != 0) {
        if (from == board.epSquare + 1 || from == board.epSquare - 1) {
          if (RANKS[from] == 4) {
            if ((BLACK_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
              board.moveBuffer[currentMoveIdx].clear();
              board.moveBuffer[currentMoveIdx].setFrom(from);
              board.moveBuffer[currentMoveIdx].setTosq(board.epSquare);
              board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
              board.moveBuffer[currentMoveIdx].setCapt(WHITE_PAWN);
              board.moveBuffer[currentMoveIdx].setEnpassant(true);
              currentMoveIdx++;
            }
          }
        }
      }

      // Pawn promotions (non-capturing)
      targetBitmap = BLACK_PAWN_MOVES[from] & ~board.occupiedSquares;
      if (targetBitmap != 0) {
        to = firstOne(targetBitmap);
        if (RANKS[to] == 1) {
          // Promotion
          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_QUEEN);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_ROOK);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_BISHOP);
          currentMoveIdx++;

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
          board.moveBuffer[currentMoveIdx].setProm(BLACK_KNIGHT);
          currentMoveIdx++;
        }
      }
    }

    // =====================================================================
    // Knights, Bishops, Rooks, Queens, King (captures only)
    // =====================================================================
    List<int> blackPieces = [
      BLACK_KNIGHT,
      BLACK_BISHOP,
      BLACK_ROOK,
      BLACK_QUEEN,
      BLACK_KING,
    ];
    List<BitMap> blackBitboards = [
      board.blackKnights,
      board.blackBishops,
      board.blackRooks,
      board.blackQueens,
      board.blackKing,
    ];

    for (int i = 0; i < blackPieces.length; i++) {
      int pieceType = blackPieces[i];
      pieceBitmap = blackBitboards[i];

      while (pieceBitmap != 0) {
        from = firstOne(pieceBitmap);
        pieceBitmap ^= BITSET[from];

        BitMap attacks;
        if (pieceType == BLACK_KNIGHT) {
          attacks = KNIGHT_ATTACKS[from];
        } else if (pieceType == BLACK_KING) {
          attacks = KING_ATTACKS[from];
        } else if (pieceType == BLACK_BISHOP) {
          attacks = getBishopAttacks(from, board.occupiedSquares);
        } else if (pieceType == BLACK_ROOK) {
          attacks = getRookAttacks(from, board.occupiedSquares);
        } else {
          // QUEEN
          attacks =
              getBishopAttacks(from, board.occupiedSquares) |
              getRookAttacks(from, board.occupiedSquares);
        }

        targetBitmap = attacks & board.whitePieces; // Only captures
        while (targetBitmap != 0) {
          to = firstOne(targetBitmap);
          targetBitmap ^= BITSET[to];

          board.moveBuffer[currentMoveIdx].clear();
          board.moveBuffer[currentMoveIdx].setFrom(from);
          board.moveBuffer[currentMoveIdx].setTosq(to);
          board.moveBuffer[currentMoveIdx].setPiec(pieceType);
          board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
          currentMoveIdx++;
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

/// Generates only pseudo-legal capture and promotion moves.
/// Translates `captgen()` from kennyMoveGen.cpp.
/// Returns the new length of the move buffer, with captures sorted by SEE.
// int captgen(int moveBufStartIdx) {
//   int currentMoveIdx = moveBufStartIdx;
//   int from, to;
//   BitMap targetBitmap;
//   BitMap pieceBitmap;

//   // Clear the move buffer for the current ply
//   board.moveBufLen[board.endOfGame + 1] = 0; // Reset next ply's start

//   // Determine the side to move
//   if (board.nextMove == WHITE_MOVE) {
//     // Generate White's captures and promotions
//     // =====================================================================
//     // Pawns
//     // =====================================================================
//     pieceBitmap = board.whitePawns;
//     while (pieceBitmap != 0) {
//       from = firstOne(pieceBitmap);
//       pieceBitmap ^= BITSET[from];

//       // Pawn captures
//       targetBitmap = WHITE_PAWN_ATTACKS[from] & board.blackPieces;
//       while (targetBitmap != 0) {
//         to = firstOne(targetBitmap);
//         targetBitmap ^= BITSET[to];

//         board.moveBuffer[currentMoveIdx].clear();
//         board.moveBuffer[currentMoveIdx].setFrom(from);
//         board.moveBuffer[currentMoveIdx].setTosq(to);
//         board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//         board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//         if (RANKS[to] == 8) {
//           // Promotion with capture
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_QUEEN);
//           currentMoveIdx++;
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_ROOK);
//           currentMoveIdx++;
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_BISHOP);
//           currentMoveIdx++;
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_KNIGHT);
//           currentMoveIdx++;
//         } else {
//           currentMoveIdx++;
//         }
//       }

//       // En passant captures
//       if (board.epSquare != 0) {
//         if (from == board.epSquare + 1 || from == board.epSquare - 1) {
//           if (RANKS[from] == 5) {
//             if ((WHITE_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
//               board.moveBuffer[currentMoveIdx].clear();
//               board.moveBuffer[currentMoveIdx].setFrom(from);
//               board.moveBuffer[currentMoveIdx].setTosq(board.epSquare);
//               board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//               board.moveBuffer[currentMoveIdx].setCapt(BLACK_PAWN);
//               board.moveBuffer[currentMoveIdx].setEnpassant(true);
//               currentMoveIdx++;
//             }
//           }
//         }
//       }

//       // Pawn promotions (non-capturing)
//       targetBitmap = WHITE_PAWN_MOVES[from] & ~board.occupiedSquares;
//       if (targetBitmap != 0) {
//         to = firstOne(targetBitmap);
//         if (RANKS[to] == 8) {
//           // Promotion
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_QUEEN);
//           currentMoveIdx++;

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_ROOK);
//           currentMoveIdx++;

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_BISHOP);
//           currentMoveIdx++;

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(WHITE_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(WHITE_KNIGHT);
//           currentMoveIdx++;
//         }
//       }
//     }

//     // =====================================================================
//     // Knights, Bishops, Rooks, Queens, King (captures only)
//     // =====================================================================
//     List<int> whitePieces = [WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING];
//     List<BitMap> whiteBitboards = [board.whiteKnights, board.whiteBishops, board.whiteRooks, board.whiteQueens, board.whiteKing];

//     for (int i = 0; i < whitePieces.length; i++) {
//       int pieceType = whitePieces[i];
//       pieceBitmap = whiteBitboards[i];

//       while (pieceBitmap != 0) {
//         from = firstOne(pieceBitmap);
//         pieceBitmap ^= BITSET[from];

//         BitMap attacks;
//         if (pieceType == WHITE_KNIGHT) {
//           attacks = KNIGHT_ATTACKS[from];
//         } else if (pieceType == WHITE_KING) {
//           attacks = KING_ATTACKS[from];
//         } else if (pieceType == WHITE_BISHOP) {
//           attacks = getBishopAttacks(from, board.occupiedSquares);
//         } else if (pieceType == WHITE_ROOK) {
//           attacks =getRookAttacks(from, board.occupiedSquares);
//         } else { // QUEEN
//           attacks = getBishopAttacks(from, board.occupiedSquares) |getRookAttacks(from, board.occupiedSquares);
//         }

//         targetBitmap = attacks & board.blackPieces; // Only captures
//         while (targetBitmap != 0) {
//           to = firstOne(targetBitmap);
//           targetBitmap ^= BITSET[to];

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(pieceType);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           currentMoveIdx++;
//         }
//       }
//     }
//   } else {
//     // Generate Black's captures and promotions (similar logic, mirrored)
//     // =====================================================================
//     // Pawns
//     // =====================================================================
//     pieceBitmap = board.blackPawns;
//     while (pieceBitmap != 0) {
//       from = firstOne(pieceBitmap);
//       pieceBitmap ^= BITSET[from];

//       // Pawn captures
//       targetBitmap = BLACK_PAWN_ATTACKS[from] & board.whitePieces;
//       while (targetBitmap != 0) {
//         to = firstOne(targetBitmap);
//         targetBitmap ^= BITSET[to];

//         board.moveBuffer[currentMoveIdx].clear();
//         board.moveBuffer[currentMoveIdx].setFrom(from);
//         board.moveBuffer[currentMoveIdx].setTosq(to);
//         board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//         board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//         if (RANKS[to] == 1) {
//           // Promotion with capture
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_QUEEN);
//           currentMoveIdx++;
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_ROOK);
//           currentMoveIdx++;
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_BISHOP);
//           currentMoveIdx++;
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_KNIGHT);
//           currentMoveIdx++;
//         } else {
//           currentMoveIdx++;
//         }
//       }

//       // En passant captures
//       if (board.epSquare != 0) {
//         if (from == board.epSquare + 1 || from == board.epSquare - 1) {
//           if (RANKS[from] == 4) {
//             if ((BLACK_PAWN_ATTACKS[from] & BITSET[board.epSquare]) != 0) {
//               board.moveBuffer[currentMoveIdx].clear();
//               board.moveBuffer[currentMoveIdx].setFrom(from);
//               board.moveBuffer[currentMoveIdx].setTosq(board.epSquare);
//               board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//               board.moveBuffer[currentMoveIdx].setCapt(WHITE_PAWN);
//               board.moveBuffer[currentMoveIdx].setEnpassant(true);
//               currentMoveIdx++;
//             }
//           }
//         }
//       }

//       // Pawn promotions (non-capturing)
//       targetBitmap = BLACK_PAWN_MOVES[from] & ~board.occupiedSquares;
//       if (targetBitmap != 0) {
//         to = firstOne(targetBitmap);
//         if (RANKS[to] == 1) {
//           // Promotion
//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_QUEEN);
//           currentMoveIdx++;

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_ROOK);
//           currentMoveIdx++;

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_BISHOP);
//           currentMoveIdx++;

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(BLACK_PAWN);
//           board.moveBuffer[currentMoveIdx].setProm(BLACK_KNIGHT);
//           currentMoveIdx++;
//         }
//       }
//     }

//     // =====================================================================
//     // Knights, Bishops, Rooks, Queens, King (captures only)
//     // =====================================================================
//     List<int> blackPieces = [BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING];
//     List<BitMap> blackBitboards = [board.blackKnights, board.blackBishops, board.blackRooks, board.blackQueens, board.blackKing];

//     for (int i = 0; i < blackPieces.length; i++) {
//       int pieceType = blackPieces[i];
//       pieceBitmap = blackBitboards[i];

//       while (pieceBitmap != 0) {
//         from = firstOne(pieceBitmap);
//         pieceBitmap ^= BITSET[from];

//         BitMap attacks;
//         if (pieceType == BLACK_KNIGHT) {
//           attacks = KNIGHT_ATTACKS[from];
//         } else if (pieceType == BLACK_KING) {
//           attacks = KING_ATTACKS[from];
//         } else if (pieceType == BLACK_BISHOP) {
//           attacks = getBishopAttacks(from, board.occupiedSquares);
//         } else if (pieceType == BLACK_ROOK) {
//           attacks =getRookAttacks(from, board.occupiedSquares);
//         } else { // QUEEN
//           attacks = getBishopAttacks(from, board.occupiedSquares) |getRookAttacks(from, board.occupiedSquares);
//         }

//         targetBitmap = attacks & board.whitePieces; // Only captures
//         while (targetBitmap != 0) {
//           to = firstOne(targetBitmap);
//           targetBitmap ^= BITSET[to];

//           board.moveBuffer[currentMoveIdx].clear();
//           board.moveBuffer[currentMoveIdx].setFrom(from);
//           board.moveBuffer[currentMoveIdx].setTosq(to);
//           board.moveBuffer[currentMoveIdx].setPiec(pieceType);
//           board.moveBuffer[currentMoveIdx].setCapt(board.square[to]);
//           currentMoveIdx++;
//         }
//       }
//     }
//   }

//   // Filter out illegal moves (moves that leave own king in check)
//   int legalMovesCount = moveBufStartIdx;
//   for (int i = moveBufStartIdx; i < currentMoveIdx; i++) {
//     Move currentMove = board.moveBuffer[i];
//     makeMove(currentMove); // Temporarily make the move
//     if (!isOwnKingAttacked()) {
//       // If king is not attacked after the move, it's legal
//       board.moveBuffer[legalMovesCount] = currentMove; // Keep the legal move
//       legalMovesCount++;
//     }
//     unmakeMove(currentMove); // Unmake the move
//   }

//   // Sort captures by SEE (Static Exchange Evaluation)
//   // The C++ code calls `addCaptScore` which uses `SEE` and sorts.
//   // This will require the `SEE` function to be implemented.
//   // For now, a simple sort by captured piece value (MVV/LVA)
//   // or a placeholder for SEE-based sorting.
//   // The C++ `captgen` function directly calls `addCaptScore` to sort.
//   // We'll sort the generated moves here.
//   for (int i = moveBufStartIdx; i < legalMovesCount; i++) {
//     // Calculate a simple MVV/LVA score for sorting captures.
//     // Higher score means better capture.
//     int score = 0;
//     Move move = board.moveBuffer[i];
//     if (move.isCapture()) {
//       score = PIECEVALUES[move.getCapt()] * 10 - PIECEVALUES[move.getPiec()];
//     } else if (move.isPromotion()) {
//       score = PIECEVALUES[move.getProm()];
//     }
//     board.moveBuffer[i].moveInt = (board.moveBuffer[i].moveInt & 0xFFFFFFFF) | (score << 32); // Store score in upper bits
//   }

//   // Sort moves in descending order of score (best captures first)
//   // Using a simple bubble sort for now, can be optimized later.
//   for (int i = moveBufStartIdx; i < legalMovesCount - 1; i++) {
//     for (int j = i + 1; j < legalMovesCount; j++) {
//       int score1 = (board.moveBuffer[i].moveInt >> 32).toInt();
//       int score2 = (board.moveBuffer[j].moveInt >> 32).toInt();
//       if (score1 < score2) {
//         Move temp = board.moveBuffer[i];
//         board.moveBuffer[i] = board.moveBuffer[j];
//         board.moveBuffer[j] = temp;
//       }
//     }
//   }

//   // Restore moveInt to original format (remove score)
//   for (int i = moveBufStartIdx; i < legalMovesCount; i++) {
//     board.moveBuffer[i].moveInt &= 0xFFFFFFFF;
//   }

//   board.moveBufLen[board.endOfGame + 1] = legalMovesCount;
//   return legalMovesCount;
// }

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

/// Helper function to get attacks for a rook from a given square, considering blockers.
/// This is a simplified version; a full magic bitboard implementation would be more complex.
/// It traces rays in rank and file directions until a blocker or board edge is hit.
BitMap getRookAttacks(int square, BitMap occupied) {
  BitMap attacks = 0;
  int r = square ~/ 8;
  int f = square % 8;

  // North (rank++)
  for (int i = r + 1; i < 8; i++) {
    int targetSq = i * 8 + f;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  // South (rank--)
  for (int i = r - 1; i >= 0; i--) {
    int targetSq = i * 8 + f;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  // East (file++)
  for (int i = f + 1; i < 8; i++) {
    int targetSq = r * 8 + i;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  // West (file--)
  for (int i = f - 1; i >= 0; i--) {
    int targetSq = r * 8 + i;
    attacks |= BITSET[targetSq];
    if ((occupied & BITSET[targetSq]) != 0) break;
  }
  return attacks;
}

/// Checks if the current side's king is attacked.
/// Translates `isOwnKingAttacked()` from kennyMoveGen.cpp.
BOOLTYPE isOwnKingAttacked() {
  int kingSquare;
  BitMap opponentPieces;

  if (board.nextMove == WHITE_MOVE) {
    kingSquare = firstOne(board.whiteKing);
    opponentPieces = board.blackPieces;
  } else {
    kingSquare = firstOne(board.blackKing);
    opponentPieces = board.whitePieces;
  }

  return isAttacked(opponentPieces, kingSquare);
}

/// Checks if the other side's king is attacked.
/// Translates `isOtherKingAttacked()` from kennyMoveGen.cpp.
BOOLTYPE isOtherKingAttacked() {
  int kingSquare;
  BitMap opponentPieces; // This will be the attacking side's pieces

  if (board.nextMove == WHITE_MOVE) {
    // If it's white's turn, we are checking if black's king is attacked by white pieces.
    kingSquare = firstOne(board.blackKing);
    opponentPieces = board.whitePieces;
  } else {
    // If it's black's turn, we are checking if white's king is attacked by black pieces.
    kingSquare = firstOne(board.whiteKing);
    opponentPieces = board.blackPieces;
  }

  return isAttacked(opponentPieces, kingSquare);
}

/// Checks if a given square is attacked by any piece of the specified color.
/// Translates `isAttacked()` from kennyFuncs.h (declaration) and `kennyMoveGen.cpp` (implementation).
/// [attackingSidePieces] A bitmask of all pieces of the attacking side.
/// [targetSquare] The square to check for attacks.
BOOLTYPE isAttacked(BitMap attackingSidePieces, int targetSquare) {
  // Check for pawn attacks
  if (board.nextMove == WHITE_MOVE) {
    // Check if any black pawn attacks targetSquare
    if ((BLACK_PAWN_ATTACKS[targetSquare] & board.blackPawns) != 0) return true;
  } else {
    // Check if any white pawn attacks targetSquare
    if ((WHITE_PAWN_ATTACKS[targetSquare] & board.whitePawns) != 0) return true;
  }

  // Check for knight attacks
  if ((KNIGHT_ATTACKS[targetSquare] &
          ((board.nextMove == WHITE_MOVE)
              ? board.blackKnights
              : board.whiteKnights)) !=
      0) {
    return true;
  }

  // Check for king attacks
  if ((KING_ATTACKS[targetSquare] &
          ((board.nextMove == WHITE_MOVE)
              ? board.blackKing
              : board.whiteKing)) !=
      0) {
    return true;
  }

  // Check for sliding piece attacks (Rooks, Queens on ranks/files; Bishops, Queens on diagonals)
  // This requires calculating attacks considering blockers.

  // Check for Rook/Queen attacks on ranks/files
  BitMap rookQueenAttackers = (board.nextMove == WHITE_MOVE)
      ? (board.blackRooks | board.blackQueens)
      : (board.whiteRooks | board.whiteQueens);
  if ((getRookAttacks(targetSquare, board.occupiedSquares) &
          rookQueenAttackers) !=
      0) {
    return true;
  }

  // Check for Bishop/Queen attacks on diagonals
  BitMap bishopQueenAttackers = (board.nextMove == WHITE_MOVE)
      ? (board.blackBishops | board.blackQueens)
      : (board.whiteBishops | board.whiteQueens);
  if ((getBishopAttacks(targetSquare, board.occupiedSquares) &
          bishopQueenAttackers) !=
      0) {
    return true;
  }

  return false;
}
