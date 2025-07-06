import 'dart:math';
import 'dart:io';

import 'package:brain_learn/utils.dart';

import 'board.dart';
import 'make_move2.dart';

import 'data.dart';
import 'move.dart';
import 'move_gen3.dart';

void main() {
  dataInit();
  board = Board();
  board.init();

  while (true) {
    final uci = stdin.readLineSync();

    if (uci == null) continue;

    if (uci == "uci") {
      print("id name Monica");
      print("id author Kelly Kinyama");
      print("uciok");
    } else if (uci == "isready") {
      print("readyok");
    } else if (uci.startsWith("position")) {
      List<String> parts = uci.split(' ');
      board.init(); // Reset to startpos

      int movesIndex = parts.indexOf("moves");
      if (movesIndex != -1) {
        List<String> movesList = parts.sublist(movesIndex + 1);
        for (String moveStr in movesList) {
          Move outMove = Move();
          // Generate legal moves to find the one that matches the string
          int end = movegen(board.moveBufLen[board.endOfGame]);
          bool moveFound = false;
          for (int i = board.moveBufLen[board.endOfGame]; i < end; i++) {
            if (board.moveBuffer[i].toAlgebraic() == moveStr) {
              outMove = board.moveBuffer[i];
              moveFound = true;
              break;
            }
          }

          if (moveFound) {
            makeMove(outMove);
          } else {
            // This indicates a problem with the UCI command or the move generator
            print("Error: Could not find or make move $moveStr");
            break;
          }
        }
      }
    } else if (uci.startsWith("go")) {
      // Generate all legal moves for the current position.
      int endIdx = movegen(board.moveBufLen[board.endOfGame]);
      int startIdx = board.moveBufLen[board.endOfGame];
      int numLegalMoves = endIdx - startIdx;

      if (numLegalMoves > 0) {
        // Select a random legal move
        final randomIndex = startIdx + Random().nextInt(numLegalMoves);
        final moveFound = board.moveBuffer[randomIndex];
        print("bestmove ${moveFound.toAlgebraic()}");
      } else {
        // No legal moves, could be checkmate or stalemate
        print("bestmove 0000");
      }
    }
  }
}
