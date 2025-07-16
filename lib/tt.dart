import 'defs.dart';

import 'move.dart';

enum NodeType { exact, lowerBound, upperBound }

class TTEntry {
  U64 key;
  Move move;
  int score;
  int depth;
  NodeType nodeType;

  TTEntry(this.key, this.move, this.score, this.depth, this.nodeType);
}

class TTable {
  TTEntry newEntry;
  TTEntry? deep;

  TTable(this.deep, this.newEntry);
}

class TT {
  Map<int, TTable> tt = {};
  final int size;

  TT(this.size);

  void addEntry(TTEntry entry) {
    int slot = entry.key % size;
    final ttEntry = tt[slot];
    if (ttEntry == null) {
      tt[slot] = TTable(null, entry);
    } else {
      if (entry.depth >= ttEntry.newEntry.depth) {
        ttEntry.deep = entry;
      } else {
        ttEntry.deep = ttEntry.newEntry;
        ttEntry.newEntry = entry;
      }
    }
  }

  TTEntry? probe(U64 key) {
    int slot = key % size;
    final ttEntry = tt[slot];

    if (ttEntry == null) return null;
    if (ttEntry.deep != null && ttEntry.deep!.key == key) {
      return ttEntry.deep;
    }
    if (ttEntry.newEntry.key == key) {
      return ttEntry.newEntry;
    }
  }
}

late TT minimaxTree;
