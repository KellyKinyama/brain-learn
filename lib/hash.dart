/// kenny_hash.dart
///
/// This file defines the `HashKeys` class, responsible for generating and storing
/// random 64-bit keys used to create an 'almost' unique signature for each
/// chess board position (Zobrist hashing).
/// It translates the C++ `HashKeys` struct from kennyHash.h and its methods from kennyHash.cpp.

import 'dart:math';
import 'defs.dart';

class HashKeys {
  // total size = 1093 * 8 = 8744 bytes (minimum required is 6312):
  // keys[64][16]: position, piece (only 12 out of 16 piece are values used)
  late List<List<U64>> keys;
  late U64 side; // side to move (black)
  late List<U64> ep; // ep targets (only 16 used)
  late U64 wk; // white king-side castling right
  late U64 wq; // white queen-side castling right
  late U64 bk; // black king-side castling right
  late U64 bq; // black queen-side castling right

  final Random _random = Random(); // Dart's Random for random number generation

  HashKeys() {
    keys = List.generate(64, (_) => List.filled(16, 0));
    ep = List.filled(64, 0);
  }

  /// Initializes all random 64-bit numbers.
  /// In C++, it uses `srand(time(NULL))` for seeding. In Dart, `Random()` is
  /// typically seeded with `DateTime.now().microsecondsSinceEpoch` for
  /// more unique seeds, or left unseeded for a default seed.
  void init() {
    for (int i = 0; i < 64; i++) {
      ep[i] = rand64();
      for (int j = 0; j < 16; j++) {
        keys[i][j] = rand64();
      }
    }
    side = rand64();
    wk = rand64();
    wq = rand64();
    bk = rand64();
    bq = rand64();
  }

  /// Generates a 64-bit random number.
  /// Dart's `Random().nextInt(2^32)` generates a 32-bit non-negative integer.
  /// To get a 64-bit number, we combine two 32-bit numbers.
  U64 rand64() {
    // Combine two 32-bit random numbers to form a 64-bit number.
    // Dart's int can hold 64-bit values.
    return (_random.nextInt(1 << 30) <<
            34) | // Shift by 34 to ensure distinct 32-bit parts
        (_random.nextInt(1 << 30) << 4) |
        _random.nextInt(1 << 4);
  }
}

// Global KEY instance (from kennyGlobals.h)
final HashKeys KEY = HashKeys();
