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
import 'move_gen3.dart'; // For captgen, isOwnKingAttacked, isOtherKingAttacked
import 'make_move2.dart'; // For makeMove, unmakeMove
import 'eval.dart'; // For eval()
import 'peek.dart';
import 'search.dart'; // For readClockAndInput (to check for timedout)

int qsearch(int ply, int alpha, int beta) {
  // quiescence search

  int i, j, val;

  // if (timedout) return 0;
  board.triangularLength[ply] = ply;
  if (isOwnKingAttacked()) return alphabetapvs(ply, 1, alpha, beta);
  val = board.eval();
  if (val >= beta) return val;
  if (val > alpha) alpha = val;

  // generate captures & promotions:
  // captgen returns a sorted move list
  board.moveBufLen[ply + 1] = captgen(board.moveBufLen[ply]);
  for (i = board.moveBufLen[ply]; i < board.moveBufLen[ply + 1]; i++) {
    makeMove(board.moveBuffer[i]);
    {
      if (board.moveBuffer[i].isCapture() ||
          board.moveBuffer[i].isPromotion() ||
          board.moveBuffer[i].isCastle()) {
      } else {
        // val = -board.eval();
        unmakeMove(board.moveBuffer[i]);
        return -board.eval();
        // continue;
      }
      if (!isOtherKingAttacked()) {
        // inodes++;
        // if (--countdown <=0) readClockAndInput();
        val = -qsearch(ply + 1, -beta, -alpha);
        unmakeMove(board.moveBuffer[i]);
        if (val >= beta) return val;
        if (val > alpha) {
          alpha = val;
          board.triangularArray[ply][ply] = board.moveBuffer[i];
          for (j = ply + 1; j < board.triangularLength[ply + 1]; j++) {
            board.triangularArray[ply][j] = board.triangularArray[ply + 1][j];
          }
          board.triangularLength[ply] = board.triangularLength[ply + 1];
        }
      } else {
        unmakeMove(board.moveBuffer[i]);
      }
    }
  }
  return alpha;
}
