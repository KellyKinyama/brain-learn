import 'dart:math' as math;
import 'package:brain_learn/defs.dart';
import 'package:chess/chess.dart';

import 'eval.dart';

import 'dart:math';

const VALUE_KNOWN_WIN = 9999.99;
double valueToReward(int value) {
  // Equivalent of `const double k = -0.00490739829861;`
  const double k = -0.00490739829861;
  double r = 1.0 / (1.0 + exp(k * value));

  // Optionally clamp reward (if you have REWARD_MATED or REWARD_MATE defined)
  r = r.clamp(0.0, 1.0);
  return r;
}

int rewardToValue(double reward) {
  // Early clamping for extreme reward values
  if (reward > 0.99) return VALUE_KNOWN_WIN.toInt();
  if (reward < 0.01) return -VALUE_KNOWN_WIN.toInt();

  const double g = 203.77396313709564; // inverse of k
  double value = g * log(reward / (1.0 - reward));
  return value.round();
}

class Edge {
  Move move;
  double? score;
  int visits = 0;
  Edge(this.move, {this.score});
}

class Node {
  U64 key;
  List<Edge>? edges;
  Node? parent;
  double? score;

  int nodeVisits = 0;
  Node(this.key, {this.edges});

  Edge? currentEdge;
}

class MCTS {
  static const size = 2;
  int exponent;
  late final int divisor;
  Map<U64, List<Node>> mctsTree = {};

  MCTS(this.exponent) {
    divisor = math.pow(size, exponent).toInt();
  }

  Node getNode(U64 key) {
    // print("Getting node: $key");
    final hashkey = key ~/ divisor;

    if (mctsTree[hashkey] == null) {
      mctsTree[hashkey] = [];
      final node = Node(key);
      mctsTree[hashkey]!.add(node);
      return node;
    } else {
      final nodes = mctsTree[hashkey];
      for (Node mctsNode in nodes!) {
        if (mctsNode.key == key) {
          return mctsNode;
        }
      }
      final node = Node(key);
      mctsTree[hashkey]!.add(node);

      return node;
    }
  }
}

MCTS mcts = MCTS(16);
