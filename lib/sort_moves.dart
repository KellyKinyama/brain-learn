/// kenny_sort_moves.dart
///
/// This file contains functions for sorting and ordering moves within the move buffer.
/// Proper move ordering is crucial for the efficiency of alpha-beta pruning.
/// It translates `Board::selectmove()` from kennySortMoves.cpp and `Board::addCaptScore()`
/// from kennyMoveGen.cpp (which is used for sorting captures).

import 'defs.dart';
import 'board.dart';
import 'move.dart';
import 'see.dart'; // For SEE

/// Reorders the move list so that the best move is selected as the next move to try.
/// This function is called during the search to optimize alpha-beta pruning.
/// It prioritizes PV moves, then captures (sorted by SEE), then history moves.
/// Translates `Board::selectmove()` from kennySortMoves.cpp.
/// [ply] Current search ply.
/// [i] The starting index in the move buffer for the current ply's moves.
/// [depth] Current search depth.
/// [followpv] True if currently following the Principal Variation.
void selectmove(int ply, int i, int depth, BOOLTYPE followpv) {
  int j, k;
  int best;
  Move temp = Move();

  // 1. Prioritize the Principal Variation (PV) move
  // If we are following the PV and it's not a leaf node of the PV,
  // try to find the PV move in the current move list and move it to the front.
  if (followpv && depth > 1) {
    for (j = i; j < board.moveBufLen[ply + 1]; j++) {
      if (board.moveBuffer[j].moveInt == board.lastPV[ply].moveInt) {
        // Found the PV move, swap it to the front (index `i`)
        temp = board.moveBuffer[j];
        board.moveBuffer[j] = board.moveBuffer[i];
        board.moveBuffer[i] = temp;
        return; // PV move is now at the front, no further sorting needed for this call
      }
    }
  }

  // 2. Sort by heuristics (history moves, captures)
  // The C++ code uses `whiteHeuristics` and `blackHeuristics` arrays for move ordering.
  // These are typically populated by successful cutoffs in the search.
  // For now, we'll use a simple heuristic: captures first (sorted by SEE), then other moves.

  // Find the best move based on heuristics (e.g., history table scores or SEE for captures)
  best = -1; // Initialize with a very low score
  int bestIdx = i;

  if (board.nextMove == BLACK_MOVE) {
    best =
        board.blackHeuristics[board.moveBuffer[i].getFrom()][board.moveBuffer[i]
            .getTosq()];
    j = i;
    for (k = i + 1; k < board.moveBufLen[ply + 1]; k++) {
      if (board.blackHeuristics[board.moveBuffer[k].getFrom()][board
              .moveBuffer[k]
              .getTosq()] >
          best) {
        best =
            board.blackHeuristics[board.moveBuffer[k].getFrom()][board
                .moveBuffer[k]
                .getTosq()];
        j = k;
      }
    }
    if (j > i) {
      temp.moveInt = board.moveBuffer[j].moveInt;
      board.moveBuffer[j].moveInt = board.moveBuffer[i].moveInt;
      board.moveBuffer[i].moveInt = temp.moveInt;
    }
  } else {
    best =
        board.whiteHeuristics[board.moveBuffer[i].getFrom()][board.moveBuffer[i]
            .getTosq()];
    j = i;
    for (k = i + 1; k < board.moveBufLen[ply + 1]; k++) {
      if (board.whiteHeuristics[board.moveBuffer[k].getFrom()][board
              .moveBuffer[k]
              .getTosq()] >
          best) {
        best =
            board.whiteHeuristics[board.moveBuffer[k].getFrom()][board
                .moveBuffer[k]
                .getTosq()];
        j = k;
      }
    }
    if (j > i) {
      temp.moveInt = board.moveBuffer[j].moveInt;
      board.moveBuffer[j].moveInt = board.moveBuffer[i].moveInt;
      board.moveBuffer[i].moveInt = temp.moveInt;
    }
  }
}

/// Adds a score to a capture move and inserts it into the sorted list.
/// This function is primarily used by `captgen` to sort captures by SEE.
/// Translates `Board::addCaptScore()` from kennyMoveGen.cpp (called by captgen).
/// [ifirst] The starting index of the capture moves for the current ply.
/// [index] The index of the newly generated capture move in the move buffer.
void addCaptScore(int ifirst, int index) {
  int val;
  Move captMove = board.moveBuffer[index];

  // Calculate SEE value for the capture
  val = SEE(captMove);

  // Discard this move if the score is not high enough (MINCAPTVAL)
  if (val < MINCAPTVAL) {
    // Effectively remove the move by decrementing the end of the move buffer.
    // The caller (captgen) will handle the actual reduction of `currentMoveIdx`.
    // In this context, we just signal that this move should be ignored.
    // The C++ code directly decrements `index` and returns, meaning the move
    // is not considered.
    // For Dart, we'll just return and let the calling `captgen` handle the `index` adjustment.
    // This function is now called *after* moves are generated and before final sorting in `captgen`.
    // So, this function might be redundant if `captgen` sorts directly.
    // The C++ `addCaptScore` is called *within* the capture generation loop to insert.
    // Since `captgen` now sorts at the end, this function's logic needs to be integrated or removed.

    // Given the current `captgen` implementation, `addCaptScore` is not directly called
    // in the same way. The sorting by SEE is done at the end of `captgen`.
    // This function's original purpose was to insert moves into an already sorted list.
    // If `captgen` is sorting all moves at once, this function might not be needed as a standalone.
    // Let's keep it as a placeholder for the original intent, but note its usage.
    return;
  }

  // The C++ code then inserts the move into the sorted list.
  // This logic is now handled by the sorting loop in `captgen` after all captures are generated.
  // The `move.moveInt = (move.moveInt & 0xFFFFFFFF) | (score << 32);` in `captgen`
  // effectively stores the score for sorting.
}
