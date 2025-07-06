import 'dart:math';
import 'dart:io';

import 'package:brain_learn/utils.dart';

import 'board.dart';
import 'make_move2.dart';

import 'data.dart';
import 'move.dart';
import 'move_gen2.dart';
// import 'perft.dart';

int maxChecksMoveCount = 20;

// void main() {
//   dataInit();
//   board = Board();
//   board.init();

//   board.display();

//   // print(perft(0, 3));

//   for (int x = 0; x < 20; x++) {
//     final movesLength = movegen(x);
//     final moves = board.moveBuffer.sublist(x, movesLength);

//     // print("move buffer: ${movegen(0)}");
//     print("Moves: $moves");
//     Move moveFound = moves[Random().nextInt(moves.length - 1).floor()];

//     makeMove(moveFound);
//     int moveCount = 0;
//     while (isOwnKingAttacked()) {
//       moveCount++;
//       if (moveCount > maxChecksMoveCount) {
//         print("Checkmate or draw");
//         return;
//       }
//       // if (isOwnKingAttacked()) {
//       unmakeMove(moveFound);
//       moveFound = moves[Random().nextInt(moves.length - 1).floor()];
//       makeMove(moveFound);
//     }

//     if (moveCount > maxChecksMoveCount) break;
//     print("Move found: $moveFound");

//     board.display();
//   }
// }

void main() {
  dataInit();
  board = Board();
  board.init();

  while (true) {
    // dataInit();

    final uci = stdin.readLineSync();
    // print("UCI command: $uci");

    switch (uci) {
      case "uci":
        {
          print("name Monica");
          print("author Kelly Kinyama");
          print("uciok");
        }

      case "isready":
        {
          print("readyok");
        }

      // case "position startpos":
      //   {}
      //position startpos moves c2c4 f7f6 b1c3 b7b6 d2d4 c8a6 e2e3 b6b5 f1d3 e7e6 d1h5
      // "go wtime 300000 btime 300000 winc 0 binc 0"
      default:
        {
          if (uci != null) {
            List<String> movesList = [];
            List<Move> moves = [];
            if (uci.startsWith("position startpos moves ")) {
              board = Board();
              board.init();
              movesList = uci.substring(24).split(" ");
              print("Parsed moves: $movesList");

              for (int x = 0; x < movesList.length; x++) {
                Move outMove = Move();
                if (isValidTextMove(movesList[x], outMove)) {
                  print("makeing move: ${outMove.toAlgebraic()}");
                  // final movesLength = movegen(x);
                  // moves = board.moveBuffer.sublist(x, movesLength);
                  // moves = board.moveBuffer.sublist(x);
                  makeMove(outMove);
                } else {
                  //   board.display();
                  //   print(moves);
                  throw "Invalid move: ${outMove.toAlgebraic()}. Move index: $x. Move index: ${movesList[x]}";
                }
              }
            } else if (uci.startsWith("position startpos")) {
              board = Board();
              board.init();
            }
            if (uci.startsWith("go")) {
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
    }
  }
}
