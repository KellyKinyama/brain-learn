import 'package:brain_learn/board.dart';

import 'defs.dart';
import 'move.dart';

class Edge {
  Move move;
  double prior = 0.0;
  double actionValue = 0.0;
  double meanActionValue = 0.0;
  int visits = 0;

  Edge(this.move);
}

class Node {
  U64 key;
  List<Edge> edges = [];
  int nodeVisits = 0;
  Node(this.key);
}

class MctsMap {
  Map<U64, Node> entry = {};
}

class MctsTree {
  Map<int, MctsMap> mcts = {};
  int size;
  MctsTree(this.size);
}

late MctsTree MCTS;

// Node getNode(Board pos) {
//   int key = pos.hashkey % MCTS.size;
//   final entries=MCTS.mcts[key];

// if(entries!=null){
//   for(MctsMap entry in entries.entry.entries.toList()){

//   }
// }
//   for(int x=MCTS.mcts[key])
// }
