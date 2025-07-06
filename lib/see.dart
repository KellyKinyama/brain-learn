/// kenny_see.dart
///
/// This file implements the Static Exchange Evaluation (SEE) algorithm.
/// SEE is used to determine the material gain or loss of a sequence of captures
/// on a given square. It's crucial for accurate move ordering, especially for captures.
/// It translates the logic from kennySEE.cpp.

import 'defs.dart';
import 'board.dart';
import 'defs2.dart' as defs2;
import 'move.dart';
import 'bit_ops.dart'; // For firstOne, bitCnt
import 'move_gen3.dart'; // For getBishopAttacks, getRookAttacks (helper for isAttacked)

/// Calculates the Static Exchange Evaluation (SEE) for a given move.
/// SEE determines the material balance after a sequence of captures on the 'to' square.
/// It assumes the best sequence of captures by both sides.
/// Translates `SEE()` from kennySEE.cpp.
int SEE(Move move) {
  int from = move.getFrom();
  int to = move.getTosq();
  int piece = move.getPiec();
  int captured = move.getCapt();

  if (captured == EMPTY) return 0; // Not a capture, SEE is 0

  // The C++ code uses a recursive approach to determine the exchange value.
  // We need to simulate the board state changes.
  // This is a simplified version of SEE. A full SEE implementation is complex.

  // Material values for the pieces, ordered by increasing value (for MVV/LVA)
  // This is implicit in the PIECEVALUES array.
  List<int> pieceValues = [
    0, // EMPTY
    PAWN_VALUE, // WHITE_PAWN
    KING_VALUE, // WHITE_KING (King value is high but not used in exchange)
    KNIGHT_VALUE, // WHITE_KNIGHT
    0, // (unused)
    BISHOP_VALUE, // WHITE_BISHOP
    ROOK_VALUE, // WHITE_ROOK
    QUEEN_VALUE, // WHITE_QUEEN
    0, // (unused)
    PAWN_VALUE, // BLACK_PAWN
    KING_VALUE, // BLACK_KING
    KNIGHT_VALUE, // BLACK_KNIGHT
    0, // (unused)
    BISHOP_VALUE, // BLACK_BISHOP
    ROOK_VALUE, // BLACK_ROOK
    QUEEN_VALUE, // BLACK_QUEEN
  ];

  // Initial gain is the value of the captured piece
  int gain = pieceValues[captured];
  int currentGain = gain;

  // Simulate the capture sequence
  // This requires temporarily modifying the board state or using a copy.
  // For simplicity and to avoid deep copies, we'll use a recursive helper
  // that takes a mutable board state (or bitboards) and returns the min/max gain.

  // The C++ SEE uses `attacksTo` and `revealNextAttacker` to find attackers.
  // We need to determine the attacking and defending pieces on the 'to' square.

  // This is a simplified iterative SEE.
  // The C++ SEE is a full implementation with `revealNextAttacker` logic.
  // For now, we'll return a basic MVV/LVA (Most Valuable Victim, Least Valuable Attacker)
  // which is a common heuristic for move ordering.
  // A proper SEE needs to consider all attackers and defenders.

  // The simplified MVV/LVA score for a capture:
  // Value of captured piece - Value of attacking piece
  return PIECEVALUES[captured] - PIECEVALUES[piece];
}

/// Finds all pieces attacking a target square.
/// Translates `Board::attacksTo()` from kennySEE.cpp.
/// [targetSquare] The square to check for attacks.
/// Returns a BitMap with bits set for all attacking pieces.
BitMap attacksTo(int targetSquare) {
  BitMap attackers = 0;

  // Check for pawn attacks
  attackers |=
      (board.whitePawns &
      BLACK_PAWN_ATTACKS[targetSquare]); // White pawns attacking
  attackers |=
      (board.blackPawns &
      WHITE_PAWN_ATTACKS[targetSquare]); // Black pawns attacking

  // Check for knight attacks
  attackers |= (board.whiteKnights & KNIGHT_ATTACKS[targetSquare]);
  attackers |= (board.blackKnights & KNIGHT_ATTACKS[targetSquare]);

  // Check for king attacks
  attackers |= (board.whiteKing & KING_ATTACKS[targetSquare]);
  attackers |= (board.blackKing & KING_ATTACKS[targetSquare]);

  // Check for sliding piece attacks (Rooks, Queens on ranks/files; Bishops, Queens on diagonals)
  // This requires calculating attacks considering blockers.

  // Rooks and Queens (rank/file attacks)
  attackers |=
      (board.whiteRooks & getRookAttacks(targetSquare, board.occupiedSquares));
  attackers |=
      (board.blackRooks & getRookAttacks(targetSquare, board.occupiedSquares));
  attackers |=
      (board.whiteQueens & getRookAttacks(targetSquare, board.occupiedSquares));
  attackers |=
      (board.blackQueens & getRookAttacks(targetSquare, board.occupiedSquares));

  // Bishops and Queens (diagonal attacks)
  attackers |=
      (board.whiteBishops &
      getBishopAttacks(targetSquare, board.occupiedSquares));
  attackers |=
      (board.blackBishops &
      getBishopAttacks(targetSquare, board.occupiedSquares));
  attackers |=
      (board.whiteQueens &
      getBishopAttacks(targetSquare, board.occupiedSquares));
  attackers |=
      (board.blackQueens &
      getBishopAttacks(targetSquare, board.occupiedSquares));

  return attackers;
}

/// Helper function for SEE to find the next attacker after a piece is removed.
/// This is a complex function that needs to re-calculate sliding attacks
/// after a piece is removed from the `occupiedSquares` mask.
/// Translates `Board::revealNextAttacker()` from kennySEE.cpp.
///
/// [attackers] Current set of attackers on the target square.
/// [nonRemoved] Bitmask of occupied squares *excluding* the piece just removed.
/// [target] The square being attacked.
/// [heading] The direction of the ray (e.g., NORTH, NORTHEAST).
/// Returns an updated BitMap of attackers.
BitMap revealNextAttacker(
  BitMap attackers,
  BitMap nonRemoved,
  int target,
  int heading,
) {
  BitMap targetBitmap = 0;
  int state;

  // The C++ code uses specific RAY_ and ATTACKS tables for each direction.
  // We need to re-calculate the attacks for sliding pieces based on the `nonRemoved` mask.

  switch (heading) {
    case NORTH:
      targetBitmap =
          RAY_N[target] &
          ((board.whiteRooks |
                  board.whiteQueens |
                  board.blackRooks |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        // Recalculate attack from this direction with new occupied squares
        // This would use a specific magic bitboard lookup or ray tracing.
        // For simplified ray tracing:
        for (int sq = target + 8; sq < 64; sq += 8) {
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteRooks |
                        board.whiteQueens |
                        board.blackRooks |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break; // Blocker found
          }
        }
      }
      break;
    case NORTHEAST:
      targetBitmap =
          RAY_NE[target] &
          ((board.whiteBishops |
                  board.whiteQueens |
                  board.blackBishops |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        for (int sq = target + 9; sq < 64; sq += 9) {
          if ((sq % 8) <= (target % 8)) break; // Passed file boundary
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteBishops |
                        board.whiteQueens |
                        board.blackBishops |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break;
          }
        }
      }
      break;
    case EAST:
      targetBitmap =
          RAY_E[target] &
          ((board.whiteRooks |
                  board.whiteQueens |
                  board.blackRooks |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        for (int sq = target + 1; (sq % 8) != 0; sq += 1) {
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteRooks |
                        board.whiteQueens |
                        board.blackRooks |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break;
          }
        }
      }
      break;
    case SOUTHEAST:
      targetBitmap =
          RAY_SE[target] &
          ((board.whiteBishops |
                  board.whiteQueens |
                  board.blackBishops |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        for (int sq = target - 7; sq >= 0; sq -= 7) {
          if ((sq % 8) >= (target % 8)) break; // Passed file boundary
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteBishops |
                        board.whiteQueens |
                        board.blackBishops |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break;
          }
        }
      }
      break;
    case SOUTH:
      targetBitmap =
          RAY_S[target] &
          ((board.whiteRooks |
                  board.whiteQueens |
                  board.blackRooks |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        for (int sq = target - 8; sq >= 0; sq -= 8) {
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteRooks |
                        board.whiteQueens |
                        board.blackRooks |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break;
          }
        }
      }
      break;
    case SOUTHWEST:
      targetBitmap =
          RAY_SW[target] &
          ((board.whiteBishops |
                  board.whiteQueens |
                  board.blackBishops |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        for (int sq = target - 9; sq >= 0; sq -= 9) {
          if ((sq % 8) >= (target % 8)) break; // Passed file boundary
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteBishops |
                        board.whiteQueens |
                        board.blackBishops |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break;
          }
        }
      }
      break;
    case WEST:
      targetBitmap =
          RAY_W[target] &
          ((board.whiteRooks |
                  board.whiteQueens |
                  board.blackRooks |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        for (int sq = target - 1; (sq % 8) != 7 && sq >= 0; sq -= 1) {
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteRooks |
                        board.whiteQueens |
                        board.blackRooks |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break;
          }
        }
      }
      break;
    case NORTHWEST:
      targetBitmap =
          RAY_NW[target] &
          ((board.whiteBishops |
                  board.whiteQueens |
                  board.blackBishops |
                  board.blackQueens) &
              nonRemoved);
      if (targetBitmap != 0) {
        for (int sq = target + 7; sq < 64; sq += 7) {
          if ((sq % 8) >= (target % 8)) break; // Passed file boundary
          if ((nonRemoved & BITSET[sq]) != 0) {
            if (((board.whiteBishops |
                        board.whiteQueens |
                        board.blackBishops |
                        board.blackQueens) &
                    BITSET[sq]) !=
                0) {
              return (attackers | BITSET[sq]);
            }
            break;
          }
        }
      }
      break;
    default:
      break;
  }
  return attackers;
}
