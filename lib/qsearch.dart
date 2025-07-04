/// kenny_qsearch.dart
///
/// This file implements the Quiescence Search (qsearch) algorithm.
/// Quiescence search is a limited-depth search that only considers "forcing" moves
/// (captures, promotions, checks) to ensure that the static evaluation function
/// is called on a "quiet" position, preventing the horizon effect.
/// It translates the logic from kennyQSearch.cpp.

import 'defs.dart';
import 'board.dart';
import 'move.dart';
import 'move_gen.dart'; // For captgen, isOwnKingAttacked, isOtherKingAttacked
import 'make_move.dart'; // For makeMove, unmakeMove
import 'eval.dart'; // For eval()
import 'peek.dart'; // For readClockAndInput (to check for timedout)

/// Performs a quiescence search.
/// [ply] Current search ply.
/// [alpha] Alpha value for alpha-beta pruning.
/// [beta] Beta value for alpha-beta pruning.
/// Returns the evaluation score from the current side's perspective.
int qsearch(int ply, int alpha, int beta) {
  // Check for timeout
  if (board.timedout) return 0; // Return 0 or a special value indicating timeout

  board.triangularLength[ply] = ply; // Update triangular PV length

  // If the king is in check, extend the search depth by one ply
  // This is a common practice in quiescence search to ensure check evasions are found.
  if (isOwnKingAttacked()) {
    // If in check, we must resolve the check.
    // This typically leads to a full search (alphabeta) for one more ply.
    // The C++ code calls `alphabetapvs` here.
    // For now, we'll just return a very low score if in check and no immediate capture resolves it,
    // or call a simplified `alphabeta` if available.
    // For a proper implementation, this would call `alphabeta` or `alphabetapvs` with depth 1.
    // Since `alphabeta` is not fully implemented yet, this will be a placeholder.
    return -LARGE_NUMBER; // Placeholder: very bad score if in check
  }

  // Stand-pat evaluation: Evaluate the current position without making any forcing moves.
  // If this score is already better than beta, we can prune.
  int standPat = eval();
  if (standPat >= beta) {
    return standPat;
  }
  if (standPat > alpha) {
    alpha = standPat;
  }

  // Generate only capture and promotion moves (forcing moves)
  int currentPlyMoveStart = board.moveBufLen[ply];
  int currentPlyMoveEnd = captgen(currentPlyMoveStart); // captgen sorts moves

  // Loop over forcing moves
  for (int i = currentPlyMoveStart; i < currentPlyMoveEnd; i++) {
    Move currentMove = board.moveBuffer[i];

    makeMove(currentMove); // Temporarily make the move
    board.inodes++; // Increment nodes searched

    // Check for timeout during move making
    if (--board.countdown <= 0) {
      readClockAndInput(); // Check for time limit and user input
    }
    if (board.timedout) {
      unmakeMove(currentMove);
      return 0; // Return on timeout
    }

    // Check if the move leaves the *other* king attacked (i.e., own king is safe)
    if (!isOwnKingAttacked()) {
      // Recursively call qsearch for the next ply
      int val = -qsearch(ply + 1, -beta, -alpha); // Negamax formulation

      unmakeMove(currentMove); // Unmake the move

      if (val >= beta) {
        return val; // Beta cutoff
      }
      if (val > alpha) {
        alpha = val; // New best move found
        // Update Principal Variation (PV)
        board.triangularArray[ply][ply] = currentMove;
        for (int j = ply + 1; j < board.triangularLength[ply + 1]; j++) {
          board.triangularArray[ply][j] = board.triangularArray[ply + 1][j];
        }
        board.triangularLength[ply] = board.triangularLength[ply + 1];
      }
    } else {
      unmakeMove(currentMove); // Unmake illegal move
    }
  }

  return alpha; // Return the best score found
}