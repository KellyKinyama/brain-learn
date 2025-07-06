/// kenny_perft.dart
///
/// This file contains the `perft` function, which is used for performance testing
/// and debugging of the move generator and `makeMove`/`unmakeMove` functions.
/// It performs a full tree search up to a specified depth and counts the number of nodes.
/// Translates `perft()` from kennyPerft.cpp.

import 'defs.dart';
import 'board.dart';
import 'move.dart';
import 'move_gen3.dart'; // For movegen, isOtherKingAttacked, isOwnKingAttacked
import 'make_move2.dart'; // For makeMove, unmakeMove

/// Performs a Perft (Performance Test) search up to a given depth.
/// This function recursively generates moves, makes them, and counts the resulting nodes.
/// It's crucial for verifying the correctness of the move generator and board update logic.
/// [ply] The current ply (depth from the root of the perft search).
/// [depth] The remaining depth to search.
/// Returns the total number of nodes (positions) reached.
U64 perft(int ply, int depth) {
  // Increment total nodes for the current position (this is a node)
  board.inodes++;

  // Base case: if depth is 0, we've reached a leaf node, count it.
  if (depth == 0) {
    return 1;
  }

  // Generate moves from this position
  // The moveBufLen[ply] is the start index for moves at this ply.
  // movegen returns the end index (exclusive).
  int currentPlyMoveStart = board.moveBufLen[ply];
  int currentPlyMoveEnd = movegen(currentPlyMoveStart);

  U64 retVal = 0; // Total nodes from this position

  // Loop over moves:
  for (int i = currentPlyMoveStart; i < currentPlyMoveEnd; i++) {
    Move currentMove = board.moveBuffer[i];

    makeMove(currentMove); // Temporarily make the move

    // Check if the move leaves the *other* king attacked (i.e., the current side's king is safe)
    // This is the legality check: if the move results in our own king being in check, it's illegal.
    // The `movegen` function already filters for legality, but this check is here for robustness
    // and consistency with the C++ perft structure.
    if (!isOwnKingAttacked()) {
      // Changed from isOtherKingAttacked() to isOwnKingAttacked() for correct logic
      // Recursively call perft for the next ply
      retVal += perft(ply + 1, depth - 1);

      // Debugging statistics for depth 1 (similar to KENNY_DEBUG_PERFT)
      // These counters are global and accumulate.
      if (depth == 1) {
        if (currentMove.isCapture()) ICAPT++;
        if (currentMove.isEnpassant()) IEP++;
        if (currentMove.isPromotion()) IPROM++;
        if (currentMove.isCastleOO()) ICASTLOO++;
        if (currentMove.isCastleOOO()) ICASTLOOO++;
        // Check if the current move results in a check on the opponent's king
        // if (isOtherKingAttacked()) ICHECK++;
      }
    }

    unmakeMove(currentMove); // Unmake the move to restore board state
  }

  return retVal;
}
