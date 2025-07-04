/// kenny_eval.dart
///
/// This file implements the `eval()` function, which is the static evaluation
/// function of the Kenny chess engine. It calculates a score for the current
/// board position from White's perspective, considering material, positional
/// bonuses, pawn structure, and king safety.
/// It translates the logic from kennyEval.cpp.

import 'dart:math';

import 'defs.dart';
import 'board.dart';
import 'bit_ops.dart'; // For bitCnt, firstOne

/// Evaluates the current board position.
/// The score is calculated from White's perspective (in centipawns)
/// and then returned from the perspective of the side to move.
/// Translates `Board::eval()` from kennyEval.cpp.
int eval() {
  int score = 0;
  int whiteKingSquare = firstOne(board.whiteKing);
  int blackKingSquare = firstOne(board.blackKing);

  // Material balance
  score += board.totalWhitePawns + board.totalWhitePieces;
  score -= board.totalBlackPawns + board.totalBlackPieces;

  // Positional bonuses (Piece-Square Tables)
  // Pawns
  BitMap tempPawns = board.whitePawns;
  while (tempPawns != 0) {
    int sq = firstOne(tempPawns);
    tempPawns ^= BITSET[sq];
    score += PAWNPOS_W[sq];
  }
  tempPawns = board.blackPawns;
  while (tempPawns != 0) {
    int sq = firstOne(tempPawns);
    tempPawns ^= BITSET[sq];
    score -= PAWNPOS_B[sq];
  }

  // Knights
  BitMap tempKnights = board.whiteKnights;
  while (tempKnights != 0) {
    int sq = firstOne(tempKnights);
    tempKnights ^= BITSET[sq];
    score += KNIGHTPOS_W[sq];
  }
  tempKnights = board.blackKnights;
  while (tempKnights != 0) {
    int sq = firstOne(tempKnights);
    tempKnights ^= BITSET[sq];
    score -= KNIGHTPOS_B[sq];
  }

  // Bishops
  BitMap tempBishops = board.whiteBishops;
  while (tempBishops != 0) {
    int sq = firstOne(tempBishops);
    tempBishops ^= BITSET[sq];
    score += BISHOPPOS_W[sq];
  }
  tempBishops = board.blackBishops;
  while (tempBishops != 0) {
    int sq = firstOne(tempBishops);
    tempBishops ^= BITSET[sq];
    score -= BISHOPPOS_B[sq];
  }

  // Rooks
  BitMap tempRooks = board.whiteRooks;
  while (tempRooks != 0) {
    int sq = firstOne(tempRooks);
    tempRooks ^= BITSET[sq];
    score += ROOKPOS_W[sq];
  }
  tempRooks = board.blackRooks;
  while (tempRooks != 0) {
    int sq = firstOne(tempRooks);
    tempRooks ^= BITSET[sq];
    score -= ROOKPOS_B[sq];
  }

  // Queens
  BitMap tempQueens = board.whiteQueens;
  while (tempQueens != 0) {
    int sq = firstOne(tempQueens);
    tempQueens ^= BITSET[sq];
    score += QUEENPOS_W[sq];
  }
  tempQueens = board.blackQueens;
  while (tempQueens != 0) {
    int sq = firstOne(tempQueens);
    tempQueens ^= BITSET[sq];
    score -= QUEENPOS_B[sq];
  }

  // King safety and positional value (midgame vs endgame)
  // Determine if it's an endgame (simplified: less than two major/minor pieces per side)
  bool endgame =
      (board.totalWhitePieces < (2 * KNIGHT_VALUE + ROOK_VALUE)) &&
      (board.totalBlackPieces < (2 * KNIGHT_VALUE + ROOK_VALUE));

  if (endgame) {
    score += KINGPOS_ENDGAME_W[whiteKingSquare];
    score -= KINGPOS_ENDGAME_B[blackKingSquare];
  } else {
    score += KINGPOS_W[whiteKingSquare];
    score -= KINGPOS_B[blackKingSquare];
  }

  // Pawn structure bonuses/penalties
  // Doubled Pawns
  // For white: Iterate through files, count pawns. If > 1, apply penalty.
  for (int file = 0; file < 8; file++) {
    int whitePawnsInFile = 0;
    int blackPawnsInFile = 0;
    for (int rank = 0; rank < 8; rank++) {
      int sq = rank * 8 + file;
      if ((board.whitePawns & BITSET[sq]) != 0) {
        whitePawnsInFile++;
      }
      if ((board.blackPawns & BITSET[sq]) != 0) {
        blackPawnsInFile++;
      }
    }
    if (whitePawnsInFile > 1)
      score -= PENALTY_DOUBLED_PAWN * (whitePawnsInFile - 1);
    if (blackPawnsInFile > 1)
      score += PENALTY_DOUBLED_PAWN * (blackPawnsInFile - 1);
  }

  // Isolated Pawns (simplified check: no friendly pawns on adjacent files)
  // This is a simplified check. A full implementation would use precomputed masks.
  for (int file = 0; file < 8; file++) {
    bool whiteIsolated = true;
    bool blackIsolated = true;
    for (int rank = 0; rank < 8; rank++) {
      int sq = rank * 8 + file;
      if ((board.whitePawns & BITSET[sq]) != 0) {
        // If white pawn in this file
        whiteIsolated = true; // Assume isolated until proven otherwise
        if (file > 0) {
          for (int r = 0; r < 8; r++) {
            if ((board.whitePawns & BITSET[r * 8 + (file - 1)]) != 0) {
              whiteIsolated = false;
              break;
            }
          }
        }
        if (file < 7 && whiteIsolated) {
          for (int r = 0; r < 8; r++) {
            if ((board.whitePawns & BITSET[r * 8 + (file + 1)]) != 0) {
              whiteIsolated = false;
              break;
            }
          }
        }
        if (whiteIsolated) score -= PENALTY_ISOLATED_PAWN;
      }
      if ((board.blackPawns & BITSET[sq]) != 0) {
        // If black pawn in this file
        blackIsolated = true;
        if (file > 0) {
          for (int r = 0; r < 8; r++) {
            if ((board.blackPawns & BITSET[r * 8 + (file - 1)]) != 0) {
              blackIsolated = false;
              break;
            }
          }
        }
        if (file < 7 && blackIsolated) {
          for (int r = 0; r < 8; r++) {
            if ((board.blackPawns & BITSET[r * 8 + (file + 1)]) != 0) {
              blackIsolated = false;
              break;
            }
          }
        }
        if (blackIsolated) score += PENALTY_ISOLATED_PAWN;
      }
    }
  }

  // Passed Pawns (simplified check: no opponent pawns in front or adjacent files)
  // This also uses precomputed masks in C++. Simplified here.
  tempPawns = board.whitePawns;
  while (tempPawns != 0) {
    int sq = firstOne(tempPawns);
    tempPawns ^= BITSET[sq];
    bool isPassed = true;
    int file = sq % 8;
    int rank = sq ~/ 8;

    // Check files: current, file-1, file+1
    for (int f = max(0, file - 1); f <= min(7, file + 1); f++) {
      for (int r = rank + 1; r < 8; r++) {
        // Check ranks in front
        if ((board.blackPawns & BITSET[r * 8 + f]) != 0) {
          isPassed = false;
          break;
        }
      }
      if (!isPassed) break;
    }
    if (isPassed) score += BONUS_PASSED_PAWN * PAWN_OWN_DISTANCE[rank];
  }

  tempPawns = board.blackPawns;
  while (tempPawns != 0) {
    int sq = firstOne(tempPawns);
    tempPawns ^= BITSET[sq];
    bool isPassed = true;
    int file = sq % 8;
    int rank = sq ~/ 8;

    for (int f = max(0, file - 1); f <= min(7, file + 1); f++) {
      for (int r = rank - 1; r >= 0; r--) {
        // Check ranks in front
        if ((board.whitePawns & BITSET[r * 8 + f]) != 0) {
          isPassed = false;
          break;
        }
      }
      if (!isPassed) break;
    }
    if (isPassed) score -= BONUS_PASSED_PAWN * PAWN_OWN_DISTANCE[7 - rank];
  }

  // Bishop pair bonus
  if (bitCnt(board.whiteBishops) >= 2) score += BONUS_BISHOP_PAIR;
  if (bitCnt(board.blackBishops) >= 2) score -= BONUS_BISHOP_PAIR;

  // Rook on open file / two rooks on open file
  for (int file = 0; file < 8; file++) {
    BitMap fileMask = 0;
    for (int r = 0; r < 8; r++) {
      fileMask |= BITSET[r * 8 + file];
    }

    bool whitePawnsInFile = (fileMask & board.whitePawns) != 0;
    bool blackPawnsInFile = (fileMask & board.blackPawns) != 0;

    if (!whitePawnsInFile && !blackPawnsInFile) {
      // Open file
      if ((fileMask & board.whiteRooks) != 0) score += BONUS_ROOK_ON_OPEN_FILE;
      if ((fileMask & board.blackRooks) != 0) score -= BONUS_ROOK_ON_OPEN_FILE;
    } else if (!whitePawnsInFile && blackPawnsInFile) {
      // Semi-open file for white
      if ((fileMask & board.whiteRooks) != 0)
        score += BONUS_ROOK_ON_OPEN_FILE ~/ 2;
    } else if (whitePawnsInFile && !blackPawnsInFile) {
      // Semi-open file for black
      if ((fileMask & board.blackRooks) != 0)
        score -= BONUS_ROOK_ON_OPEN_FILE ~/ 2;
    }
  }

  // King shield (pawns in front of king)
  // This is also a simplified version. C++ uses precomputed KINGSHIELD masks.
  // White King Shield
  int whiteKingFile = whiteKingSquare % 8;
  int whiteKingRank = whiteKingSquare ~/ 8;
  int whiteShieldBonus = 0;
  if (whiteKingRank < 2) {
    // King on 1st or 2nd rank
    for (
      int f = max(0, whiteKingFile - 1);
      f <= min(7, whiteKingFile + 1);
      f++
    ) {
      if ((board.whitePawns & BITSET[BOARDINDEX[f + 1][whiteKingRank + 1]]) !=
          0) {
        whiteShieldBonus += BONUS_PAWN_SHIELD_STRONG;
      }
      if (whiteKingRank + 2 <= 8 &&
          (board.whitePawns & BITSET[BOARDINDEX[f + 1][whiteKingRank + 2]]) !=
              0) {
        whiteShieldBonus += BONUS_PAWN_SHIELD_WEAK;
      }
    }
  }
  score += whiteShieldBonus;

  // Black King Shield
  int blackKingFile = blackKingSquare % 8;
  int blackKingRank = blackKingSquare ~/ 8;
  int blackShieldBonus = 0;
  if (blackKingRank > 5) {
    // King on 7th or 8th rank
    for (
      int f = max(0, blackKingFile - 1);
      f <= min(7, blackKingFile + 1);
      f++
    ) {
      if ((board.blackPawns & BITSET[BOARDINDEX[f + 1][blackKingRank - 1]]) !=
          0) {
        blackShieldBonus += BONUS_PAWN_SHIELD_STRONG;
      }
      if (blackKingRank - 2 >= 1 &&
          (board.blackPawns & BITSET[BOARDINDEX[f + 1][blackKingRank - 2]]) !=
              0) {
        blackShieldBonus += BONUS_PAWN_SHIELD_WEAK;
      }
    }
  }
  score -= blackShieldBonus; // Subtract for black

  // Return score from the perspective of the side to move
  if (board.nextMove == BLACK_MOVE) {
    return -score;
  }
  return score;
}
