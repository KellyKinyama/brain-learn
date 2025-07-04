import 'dart:math';

import 'board.dart';
import 'make_move2.dart';

import 'data.dart';
import 'move.dart';
import 'move_gen.dart';
import 'perft.dart';

int maxChecksMoveCount = 20;

void main() {
  dataInit();
  board = Board();
  board.init();

  board.display();

  // print(perft(0, 3));

  for (int x = 0; x < 20; x++) {
    final movesLength = movegen(x);
    final moves = board.moveBuffer.sublist(x, movesLength);

    // print("move buffer: ${movegen(0)}");
    print("Moves: $moves");
    Move moveFound = moves[Random().nextInt(moves.length - 1).floor()];

    makeMove(moveFound);
    int moveCount = 0;
    while (isOwnKingAttacked()) {
      moveCount++;
      if (moveCount > maxChecksMoveCount) {
        print("Checkmate or draw");
        return;
      }
      // if (isOwnKingAttacked()) {
      unmakeMove(moveFound);
      moveFound = moves[Random().nextInt(moves.length - 1).floor()];
      makeMove(moveFound);
    }

    if (moveCount > maxChecksMoveCount) break;
    print("Move found: $moveFound");

    board.display();
  }
}
