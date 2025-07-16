/// kenny_search.dart
///
/// This file contains the main search algorithms of the Kenny chess engine,
/// including the iterative deepening framework, alpha-beta pruning,
/// and principal variation search (PVS).
/// It translates the logic from kennySearch.cpp.

import 'package:brain_learn/tt.dart';

import 'defs.dart';
import 'board.dart';
import 'hash.dart';
import 'move.dart';
import 'move_gen3.dart'; // For movegen, isOwnKingAttacked, isOtherKingAttacked
import 'make_move2.dart'; // For makeMove, unmakeMove
import 'eval.dart'; // For eval()
import 'peek.dart'; // For readClockAndInput()
import 'qsearch.dart';
import 'sort_moves.dart'; // For selectmove()

/// The main entry point for the engine's thinking process.
/// It drives iterative deepening and calls the alpha-beta search.
/// The search stops based on time limits, depth limits, or user interruption.
/// Translates `Board::think()` from kennySearch.cpp.
/// Returns the best move found.
// Move think() {
//   int score = 0;
//   int legalMoves = 0;
//   Move singleMove = NOMOVE; // Used if only one legal move
//   Move bestMove = NOMOVE;

//   board.timedout = false;
//   board.inodes = 0;
//   board.lastPVLength = 0;
//   board.followpv = false; // Start with no PV to follow
//   board.allownull = true; // Allow null moves by default

//   // Start the timer for the search
//   board.timer.init();
//   board.msStart = board.timer.getsysms();

//   // Check if the game has ended or if there is only one legal move.
//   // If so, no need to search.
//   // This calls `movegen` to get all legal moves and checks `isEndOfgame`.
//   legalMoves = movegen(
//     board.moveBufLen[board.endOfGame],
//   ); // Generate moves for ply 0 (root)
//   board.triangularArray = List.generate(
//     MAX_PLY,
//     (_) => List.generate(MAX_PLY, (__) => Move()),
//   );
//   // if (legalMoves == 0) {
//   //   // Checkmate or Stalemate
//   //   if (isOwnKingAttacked()) {
//   //     print("Checkmate!");
//   //     board.endOfGame = CHECKMATESCORE; // Or some other indicator
//   //   } else {
//   //     print("Stalemate!");
//   //     board.endOfGame = STALEMATESCORE;
//   //   }
//   //   return NOMOVE; // No legal moves
//   // } else if (legalMoves == 1) {
//   //   // Only one legal move, just play it
//   //   singleMove = board.moveBuffer[board.moveBufLen[0]];
//   //   print("Only one legal move: ${singleMove.toString()}");
//   //   // makeMove(singleMove); // Make the move directly
//   //   return singleMove;
//   // }

//   // Iterative Deepening Loop
//   for (
//     int currentDepth = 1;
//     currentDepth <= board.searchDepth;
//     currentDepth++
//   ) {
//     // Reset timeout flag for each iteration
//     board.timedout = false;
//     board.countdown = UPDATEINTERVAL; // Reset countdown for input/time checks

//     // Set alpha-beta window for aspiration windows (optional, but common)
//     // For simplicity, start with full window.
//     int alpha = -LARGE_NUMBER;
//     int beta = LARGE_NUMBER;

//     // Call the main search function (alpha-beta or PVS)
//     // The C++ code uses `alphabetapvs`.
//     score = alphabetapvs(0, currentDepth, alpha, beta);

//     // If timed out, break the iterative deepening loop
//     if (board.timedout) {
//       print("Search timed out at depth ${currentDepth - 1}.");
//       break;
//     }

//     // Update best move and PV if the search was successful (not timed out)
//     if (!board.timedout) {
//       bestMove = board.triangularArray[0][0]; // Best move from the root PV
//       board.lastPVLength = board.triangularLength[0];
//       for (int i = 0; i < board.lastPVLength; i++) {
//         board.lastPV[i] = board.triangularArray[0][i];
//       }
//     }

//     // Display search stats
//     displaySearchStats(
//       1,
//       currentDepth,
//       score,
//     ); // Mode 1 for iterative deepening

//     // Check if we should stop searching (e.g., time limit, mate found)
//     // The C++ code checks `STOPFRAC` and `CHECKMATESCORE`.
//     board.msStop = board.timer.getsysms();
//     if ((board.msStop - board.msStart) > (board.maxTime * STOPFRAC)) {
//       print("Stopping search: Time limit fraction reached.");
//       break;
//     }
//     // if (score.abs() >= CHECKMATESCORE - MAX_PLY) {
//     //   // Mate found or very high score indicating forced mate, stop early.
//     //   print("Stopping search: Mate found or forced mate detected.");
//     //   break;
//     // }
//   }

//   board.timer.stop(); // Stop the timer

//   // Make the best move found by the engine
//   // if (bestMove.moveInt != 0) {
//   //   makeMove(bestMove);
//   // } else {
//   //   // If no best move found (e.g., timed out at depth 0 or no legal moves),
//   //   // try to make the first legal move if available.
//   //   if (legalMoves > 0) {
//   //     bestMove = board.moveBuffer[board.moveBufLen[0]];
//   //     makeMove(bestMove);
//   //   }
//   // }

//   return bestMove;
// }
Move think() {
  // Generate all legal moves for the current position.
  // int endIdx = movegen(board.moveBufLen[board.endOfGame]);
  // int startIdx = board.moveBufLen[board.endOfGame];
  // int numLegalMoves = endIdx - startIdx;
  board.followpv = true;
  // int bestMoveScore = -LARGE_NUMBER; // Initialize best move score
  // Move bestMove = NOMOVE; // Initialize best move

  for (int id = 1; id <= 8; id++) {
    board.followpv = true;
    board.moveBuffer = List.generate(MAX_MOV_BUFF, (index) => Move());
    board.moveBufLen = List.filled(MAX_PLY, 0);
    board.triangularLength = List.filled(MAX_PLY, 0);
    board.triangularArray = List.generate(
      MAX_PLY,
      (_) => List.generate(MAX_PLY, (__) => Move()),
    );
    alphabetapvs(0, id, -LARGE_NUMBER, LARGE_NUMBER);
    rememberPV();
  }
  return board.triangularArray[0][0];
  // return bestMove; // Return the best move found
}

int alphabetapvs(int ply, int depth, int alpha, int beta) {
  if (depth <= 0 && !isOwnKingAttacked()) {
    board.followpv = false;
    return eval(); // Call evaluation function at leaf nodes
    // qsearch(ply, alpha, beta);
  }
  int i;
  Move? ttMove;
  // Generate all legal moves for the current posi

  // int bestMoveScore = -LARGE_NUMBER; // Initialize best move score
  // Move bestMove = NOMOVE; // Initialize best move
  // int movesfound = 0;

  board.triangularLength[ply] = ply;

  final ttEntry = minimaxTree.probe(board.hashkey);
  if (ttEntry != null && ttEntry.key == board.hashkey) {
    ttMove = ttEntry.move;
    if (ttEntry.depth >= depth && !board.followpv) {
      if (ttEntry.nodeType == NodeType.exact) {
        return ttEntry.score;
      }
      if (ttEntry.nodeType == NodeType.lowerBound && ttEntry.score >= beta) {
        // print("TT entry: $ttEntry");
        return ttEntry.score;
      }
      if (ttEntry.nodeType == NodeType.upperBound && ttEntry.score <= alpha) {
        // print("TT entry: $ttEntry");
        return ttEntry.score;
      }
    }
  }

  if (!board.followpv && board.allownull) {
    if ((board.nextMove == BLACK_MOVE &&
            (board.totalBlackPieces > NULLMOVE_LIMIT)) ||
        (board.nextMove == WHITE_MOVE &&
            (board.totalWhitePieces > NULLMOVE_LIMIT))) {
      if (!isOwnKingAttacked()) {
        board.allownull = false;
        // inodes++;
        // if (--countdown <=0) readClockAndInput();
        board.nextMove = board.nextMove == WHITE_MOVE ? BLACK_MOVE : WHITE_MOVE;
        board.hashkey ^= KEY.side;
        int val = -alphabetapvs(
          ply,
          depth - NULLMOVE_REDUCTION,
          -beta,
          -beta + 1,
        );
        board.nextMove = board.nextMove == WHITE_MOVE ? BLACK_MOVE : WHITE_MOVE;
        board.hashkey ^= KEY.side;
        // if (timedout) return 0;
        board.allownull = true;
        if (val >= beta) return val;
      }
    }
  }
  board.allownull = true;

  board.moveBufLen[ply + 1] = movegen(board.moveBufLen[ply]);
  int movesfound = 0;
  int pvmovesfound = 0;

  for (i = board.moveBufLen[ply]; i < board.moveBufLen[ply + 1]; i++) {
    selectmove(ply, i, depth, board.followpv, ttMove);
    Move move = board.moveBuffer[i];
    if (ply == 0) {
      // print("current move: ${move.toAlgebraic()}");
    }

    if (ttMove != null && move.moveInt == ttMove.moveInt) ttMove = null;
    makeMove(move);

    if (!isOwnKingAttacked()) {
      movesfound++; // Count legal moves
      // Evaluate the move
      int val = -alphabetapvs(ply + 1, depth - 1, -beta, -alpha);
      unmakeMove(move); // Unmake the move

      if (val >= beta) {
        if (board.nextMove == BLACK_MOVE) {
          board.blackHeuristics[board.moveBuffer[i].getFrom()][board
                  .moveBuffer[i]
                  .getTosq()] +=
              depth * depth;
        } else {
          board.whiteHeuristics[board.moveBuffer[i].getFrom()][board
                  .moveBuffer[i]
                  .getTosq()] +=
              depth * depth;
        }
        minimaxTree.addEntry(
          TTEntry(board.hashkey, move, beta, depth, NodeType.lowerBound),
        );
        return beta;
      }

      if (val > alpha) {
        pvmovesfound++;
        alpha = val; // Update best move score
        board.triangularArray[ply][ply] = move;
        for (int j = ply + 1; j < board.triangularLength[ply + 1]; j++) {
          board.triangularArray[ply][j] = board.triangularArray[ply + 1][j];
        }
        board.triangularLength[ply] = board.triangularLength[ply + 1];

        if (ply == 0) {
          print("current best move: ${move.toAlgebraic()}");
        }
        minimaxTree.addEntry(
          TTEntry(board.hashkey, move, val, depth, NodeType.lowerBound),
        );
      }
    } else {
      unmakeMove(move); // Unmake illegal move
    }
  }

  if (pvmovesfound > 0) {
    if (board.nextMove == BLACK_MOVE) {
      board.blackHeuristics[board.triangularArray[ply][ply].getFrom()][board
              .triangularArray[ply][ply]
              .getTosq()] +=
          depth * depth;
    } else {
      board.whiteHeuristics[board.triangularArray[ply][ply].getFrom()][board
              .triangularArray[ply][ply]
              .getTosq()] +=
          depth * depth;
    }
    minimaxTree.addEntry(
      TTEntry(board.hashkey, NOMOVE, alpha, depth, NodeType.exact),
    );
  } else {
    minimaxTree.addEntry(
      TTEntry(board.hashkey, NOMOVE, alpha, depth, NodeType.upperBound),
    );
  }

  //	Checkmate/stalemate detection:
  if (movesfound == 0) {
    if (isOwnKingAttacked()) {
      return (-CHECKMATESCORE + ply - 1);
    } else {
      return (STALEMATESCORE);
    }
  }

  return alpha;
}

/// Performs an Alpha-Beta search with Principal Variation Search (PVS) optimization.
/// Translates `Board::alphabetapvs()` from kennySearch.cpp.
/// [ply] Current search ply.
/// [depth] Remaining search depth.
/// [alpha] Alpha value for alpha-beta pruning.
/// [beta] Beta value for alpha-beta pruning.
/// Returns the evaluation score from the current side's perspective.
// int alphabetapvs(int ply, int depth, int alpha, int beta) {
//   // Check for timeout
//   if (board.timedout) return 0;

//   board.inodes++; // Increment nodes searched

//   // Base case: if depth is 0, call quiescence search.
//   if (depth == 0) {
//     // return qsearch(ply, alpha, beta);
//     return eval();
//   }

//   // Check for repetition (draw)
//   if (board.repetitionCount() >= 2) {
//     // 2 repetitions for draw
//     return DRAWSCORE;
//   }

//   // Null move pruning (if allowed and not in check)
//   // This is an optimization where we skip a move for the current side
//   // and search the opponent's response at a reduced depth.
//   // if (board.allownull &&
//   //     !isOwnKingAttacked() &&
//   //     board.Material.abs() > NULLMOVE_LIMIT) {
//   //   // Make a null move (pass the turn)
//   //   board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;
//   //   board.hashkey ^= KEY.side; // Update hash for side to move
//   //   board.epSquare = 0; // Null move clears EP square
//   //   board.hashkey ^=
//   //       KEY.ep[board.epSquare]; // Update hash for EP square (if it was set)

//   //   board.allownull = false; // Don't allow consecutive null moves
//   //   int val = -alphabetapvs(
//   //     ply + 1,
//   //     depth - NULLMOVE_REDUCTION - 1,
//   //     -beta,
//   //     -beta + 1,
//   //   );
//   //   board.allownull = true; // Restore null move allowance

//   //   // Unmake the null move
//   //   board.nextMove = (board.nextMove == WHITE_MOVE) ? BLACK_MOVE : WHITE_MOVE;
//   //   board.hashkey ^= KEY.side;
//   //   // Restore epSquare (from gameLine, if needed, or by re-calculating)
//   //   // For now, just reset it to 0 as it was cleared.
//   //   board.epSquare = board
//   //       .gameLine[board.endOfGame - 1]
//   //       .epSquare; // Restore from previous record
//   //   board.hashkey ^= KEY.ep[board.epSquare]; // Update hash for EP square

//   //   if (val >= beta) {
//   //     return beta; // Null move cutoff
//   //   }
//   // }

//   // Generate all legal moves for the current position
//   // print("Current ply: $ply, depth: $depth");
//   int currentPlyMoveStart = board.moveBufLen[ply];
//   int currentPlyMoveEnd = movegen(currentPlyMoveStart);

//   // If no legal moves, it's checkmate or stalemate
//   if (currentPlyMoveEnd == currentPlyMoveStart) {
//     return isOwnKingAttacked() ? -CHECKMATESCORE + ply : STALEMATESCORE;
//   }

//   // Initialize PV for current ply
//   board.triangularLength[ply] = ply;

//   bool firstMove = true;
//   for (int i = currentPlyMoveStart; i < currentPlyMoveEnd; i++) {
//     // Select (order) the moves
//     selectmove(ply, i, depth, board.followpv);
//     Move currentMove = board.moveBuffer[i];

//     makeMove(currentMove); // Temporarily make the move

//     if (isOwnKingAttacked()) {
//       // If the move leaves the own king attacked, skip it
//       unmakeMove(currentMove);
//       continue;
//     }

//     // Check for timeout during move making
//     if (--board.countdown <= 0) {
//       readClockAndInput();
//     }
//     if (board.timedout) {
//       unmakeMove(currentMove);
//       return 0; // Return on timeout
//     }

//     int val;
//     if (firstMove) {
//       // Full window search for the first move (PV candidate)
//       val = -alphabetapvs(ply + 1, depth - 1, -beta, -alpha);
//       firstMove = false;
//     } else {
//       // Zero-window search (scout search) for subsequent moves
//       // If this search fails high, we re-search with a full window.
//       val = -alphabetapvs(
//         ply + 1,
//         depth - 1,
//         -alpha - 1,
//         -alpha,
//       ); // Test if it's better than alpha
//       if (val > alpha && val < beta) {
//         // If it was a "fail high" (i.e., better than alpha), re-search with full window
//         val = -alphabetapvs(ply + 1, depth - 1, -beta, -alpha);
//       }
//     }

//     unmakeMove(currentMove); // Unmake the move

//     // Alpha-beta pruning logic
//     if (val >= beta) {
//       // Beta cutoff: This move is too good, opponent won't let us get here.
//       // Store in history heuristic (if non-capture)
//       if (!currentMove.isCapture()) {
//         if (board.nextMove == WHITE_MOVE) {
//           board.whiteHeuristics[currentMove.getFrom()][currentMove.getTosq()] +=
//               depth * depth;
//         } else {
//           board.blackHeuristics[currentMove.getFrom()][currentMove.getTosq()] +=
//               depth * depth;
//         }
//       }
//       return val;
//     }

//     if (val > alpha) {
//       // Found a new best move for the current side.
//       alpha = val;
//       // Update Principal Variation (PV)
//       board.triangularArray[ply][ply] = currentMove;
//       for (int j = ply + 1; j < board.triangularLength[ply + 1]; j++) {
//         board.triangularArray[ply][j] = board.triangularArray[ply + 1][j];
//       }
//       board.triangularLength[ply] = board.triangularLength[ply + 1];
//     }
//   }

//   return alpha; // Return the best score found
// }

/// Placeholder for minimax() function.
/// Translates Board::minimax() from kennySearch.cpp (if used, often replaced by alphabeta).
/// Minimax is a basic search algorithm, usually optimized with Alpha-Beta.
/// This function is likely a helper or an older version of the search.
/// Not fully translated as PVS is the primary search.
int minimax(int ply, int depth) {
  // Check for timeout
  if (board.timedout) return 0;

  board.inodes++;

  if (depth == 0) {
    return eval(); // Evaluate leaf node
  }

  // Generate all legal moves
  int currentPlyMoveStart = board.moveBufLen[ply];
  int currentPlyMoveEnd = movegen(currentPlyMoveStart);

  if (currentPlyMoveEnd == currentPlyMoveStart) {
    return isOwnKingAttacked() ? -CHECKMATESCORE + ply : STALEMATESCORE;
  }

  int bestValue = -LARGE_NUMBER; // For maximizing player (White)
  if (board.nextMove == BLACK_MOVE) {
    bestValue = LARGE_NUMBER; // For minimizing player (Black)
  }

  for (int i = currentPlyMoveStart; i < currentPlyMoveEnd; i++) {
    Move currentMove = board.moveBuffer[i];
    makeMove(currentMove);

    if (--board.countdown <= 0) {
      readClockAndInput();
    }
    if (board.timedout) {
      unmakeMove(currentMove);
      return 0;
    }

    int val = minimax(ply + 1, depth - 1); // Recursive call

    unmakeMove(currentMove);

    if (board.nextMove == WHITE_MOVE) {
      // Maximizing player
      if (val > bestValue) {
        bestValue = val;
      }
    } else {
      // Minimizing player
      if (val < bestValue) {
        bestValue = val;
      }
    }
  }
  return bestValue;
}

/// Placeholder for alphabeta() function.
/// Translates Board::alphabeta() from kennySearch.cpp.
/// This is a standard alpha-beta search. The engine primarily uses PVS.
/// Not fully translated as PVS is the primary search.
int alphabeta(int ply, int depth, int alpha, int beta) {
  // Check for timeout
  if (board.timedout) return 0;

  board.inodes++;

  if (depth == 0) {
    // return qsearch(ply, alpha, beta); // Call quiescence search at leaf nodes
    return eval();
  }

  // Check for repetition (draw)
  if (board.repetitionCount() >= 2) {
    return DRAWSCORE;
  }

  int currentPlyMoveStart = board.moveBufLen[ply];
  int currentPlyMoveEnd = movegen(currentPlyMoveStart);

  if (currentPlyMoveEnd == currentPlyMoveStart) {
    return isOwnKingAttacked() ? -CHECKMATESCORE + ply : STALEMATESCORE;
  }

  for (int i = currentPlyMoveStart; i < currentPlyMoveEnd; i++) {
    Move currentMove = board.moveBuffer[i];
    makeMove(currentMove);

    if (--board.countdown <= 0) {
      readClockAndInput();
    }
    if (board.timedout) {
      unmakeMove(currentMove);
      return 0;
    }

    int val = -alphabeta(
      ply + 1,
      depth - 1,
      -beta,
      -alpha,
    ); // Negamax formulation

    unmakeMove(currentMove);

    if (val >= beta) {
      return val; // Beta cutoff
    }
    if (val > alpha) {
      alpha = val; // New best move found
    }
  }
  return alpha; // Return the best score found
}

/// Displays search statistics.
/// Translates `Board::displaySearchStats()` from kennySearch.cpp.
/// [mode] Display mode (e.g., 1 for iterative deepening info).
/// [depth] Current search depth.
/// [score] Score of the best move found.
void displaySearchStats(int mode, int depth, int score) {
  if (mode == 1) {
    // Iterative deepening output
    String scoreStr;
    if (score > CHECKMATESCORE - MAX_PLY) {
      scoreStr = "mate ${((CHECKMATESCORE - score) / 2).ceil()}";
    } else if (score < -CHECKMATESCORE + MAX_PLY) {
      scoreStr = "mate ${((-CHECKMATESCORE - score) / 2).ceil()}";
    } else {
      scoreStr = "cp $score";
    }

    String pvString = "";
    for (int i = 0; i < board.lastPVLength; i++) {
      // Need `toSan` for proper display
      // For now, use `toString` from Move class
      pvString += "${board.lastPV[i].toString()} ";
    }

    print(
      "info depth $depth score $scoreStr nodes ${board.inodes} time ${board.timer.getms()} pv $pvString",
    );
  }
  // Add other display modes if necessary
}

/// Checks if the game has ended (checkmate, stalemate, draw).
/// Translates `Board::isEndOfgame()` from kennySearch.cpp.
/// [legalmoves] Number of legal moves from the current position.
/// [singlemove] The single legal move if only one exists.
/// Returns true if the game has ended, false otherwise.
BOOLTYPE isEndOfgame(int legalmoves, Move singlemove) {
  if (legalmoves == 0) {
    return true; // Checkmate or Stalemate
  }
  // Check for 50-move rule draw
  if (board.fiftyMove >= 100) {
    // 100 half-moves = 50 full moves
    return true;
  }
  // Check for three-fold repetition draw
  if (board.repetitionCount() >= 2) {
    return true;
  }
  return false;
}

/// Counts the number of times the current position has been repeated in the game history.
/// Translates `Board::repetitionCount()` from kennySearch.cpp.
/// Returns the count of repetitions.
int repetitionCount() {
  int count = 0;
  // Iterate through the gameLine (history) and compare hash keys.
  // The C++ code checks from `endOfGame - 2` down to `0`, skipping every other ply.
  // It also considers the 50-move rule limit.
  for (
    int i = board.endOfGame - 2;
    i >= 0 && board.gameLine[i].fiftyMove < board.fiftyMove;
    i -= 2
  ) {
    if (board.gameLine[i].key == board.hashkey) {
      count++;
    }
  }
  return count;
}

void rememberPV() {
  // remember the last PV, and also the 5 previous ones because
  // they usually contain good moves to try
  int i;
  board.lastPVLength = board.triangularLength[0];
  for (i = 0; i < board.triangularLength[0]; i++) {
    board.lastPV[i] = board.triangularArray[0][i];
  }
}
