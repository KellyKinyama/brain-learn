/// kenny_bit_ops.dart
///
/// This file contains utility functions for bitboard manipulation,
/// translating the functions from kennyBitOps.cpp.
/// These functions are essential for efficient chess engine operations.

import 'defs.dart'; // For BitMap and U64 typedefs, and MS1BTABLE

/// Counts the number of set bits (1s) in a 64-bit bitmap.
/// Translates `bitCnt()` from kennyBitOps.cpp (MIT HAKMEM algorithm).
///
/// This implementation uses Dart's built-in `bitLength` property, which
/// effectively counts the bits for positive integers. For a true population
/// count (number of set bits), we need a different approach as `bitLength`
/// gives the number of bits required to represent the integer.
///
/// A more direct translation of the HAKMEM algorithm or using `toRadixString(2)`
/// and counting '1's would be more accurate for population count.
///
/// Re-implementing the HAKMEM algorithm for clarity and direct translation:
/// static const U64  M1 = 0x5555555555555555;  // 1 zero,  1 one ...
/// static const U64  M2 = 0x3333333333333333;  // 2 zeros,  2 ones ...
/// static const U64  M4 = 0x0f0f0f0f0f0f0f0f;  // 4 zeros,  4 ones ...
/// static const U64  M8 = 0x00ff00ff00ff00ff;  // 8 zeros,  8 ones ...
/// static const U64 M16 = 0x0000ffff0000ffff;
/// static const U64 M32 = 0x00000000ffffffff;
int bitCnt(BitMap bitmap) {
  // Ensure the bitmap is treated as unsigned 64-bit.
  // Dart's `int` handles 64-bit signed, but bitwise operations work as expected.
  // For consistency with C++ unsigned behavior, we might need to handle negative
  // numbers if they arise from bitwise operations, but for positive bitmaps, it's fine.

  // The HAKMEM algorithm for popcount:
  bitmap = (bitmap & 0x5555555555555555) + ((bitmap >> 1) & 0x5555555555555555);
  bitmap = (bitmap & 0x3333333333333333) + ((bitmap >> 2) & 0x3333333333333333);
  bitmap = (bitmap + (bitmap >> 4)) & 0x0f0f0f0f0f0f0f0f;
  bitmap = bitmap + (bitmap >> 8);
  bitmap = bitmap + (bitmap >> 16);
  bitmap = bitmap + (bitmap >> 32);
  return (bitmap & 0x7F).toInt(); // The result is in the lowest 7 bits
}

/// Finds the index of the least significant set bit (first one) in a bitmap.
/// Translates `firstOne()` from kennyBitOps.cpp (MIT HAKMEM algorithm).
/// Returns 64 if bitmap is 0.
int firstOne(BitMap bitmap) {
  if (bitmap == 0) return 64; // No set bit

  // The C++ code uses a lookup table and specific bit manipulation.
  // Dart's `lowBit` (or `lowestSetBit`) is not directly available.
  // A common way to find the LSB index is `(bitmap & -bitmap)`.
  // Then, find the position of this single set bit.

  // A direct bitwise approach (similar to `__builtin_ctzll` or `_BitScanForward64`):
  int count = 0;
  if (bitmap == 0) return 64; // No set bit

  if ((bitmap & 0xFFFFFFFF) == 0) {
    bitmap >>= 32;
    count += 32;
  }
  if ((bitmap & 0xFFFF) == 0) {
    bitmap >>= 16;
    count += 16;
  }
  if ((bitmap & 0xFF) == 0) {
    bitmap >>= 8;
    count += 8;
  }
  if ((bitmap & 0xF) == 0) {
    bitmap >>= 4;
    count += 4;
  }
  if ((bitmap & 0x3) == 0) {
    bitmap >>= 2;
    count += 2;
  }
  if ((bitmap & 0x1) == 0) {
    bitmap >>= 1;
    count += 1;
  }
  return count;
}

/// Finds the index of the most significant set bit (last one) in a bitmap.
/// Translates `lastOne()` from kennyBitOps.cpp (Eugene Nalimov's bitScanReverse).
/// Returns 64 if bitmap is 0.
int lastOne(BitMap bitmap) {
  if (bitmap == 0) return 64; // No set bit

  // The C++ code uses a lookup table (MS1BTABLE) and specific bit manipulation.
  // This is a direct translation of that logic.
  int result = 0;
  if (bitmap > 0xFFFFFFFF) {
    bitmap >>= 32;
    result = 32;
  }
  if (bitmap > 0xFFFF) {
    bitmap >>= 16;
    result += 16;
  }
  if (bitmap > 0xFF) {
    bitmap >>= 8;
    result += 8;
  }
  return result + MS1BTABLE[bitmap.toInt()];
}
