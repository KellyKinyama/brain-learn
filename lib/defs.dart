/// kenny_defs.dart
///
/// This file contains the core definitions and constants used throughout the Kenny chess engine.
/// It translates the C++ macros and global constants from kennyDefs.h and parts of kennyGlobals.h
/// into Dart equivalents.

/// Typedefs for C++ types
/// In Dart, `int` can represent 64-bit integers on most platforms, making it suitable for U64 and BitMap.
/// For very large numbers or strict unsigned behavior, `BigInt` might be considered, but for bitwise
/// operations within typical 64-bit ranges, `int` is usually sufficient.
typedef U64 = int;
typedef BitMap = int;
typedef SHORTINT = int;
typedef USHORTINT = int;
typedef BOOLTYPE =
    bool; // C++ BOOLTYPE is int, Dart uses bool for boolean logic.

/// Definitions from kennyDefs.h
const String KENNY_PROG_VERSION =
    "Kenny v0.1.1, by Kenshin Himura, Copyright 2012";

const int MAX_CMD_BUFF = 256; // Console command input buffer
const int MAX_MOV_BUFF =
    4096; // Max number of moves that we can store (all plies)
const int MAX_PLY = 64; // Max search depth
const int MAX_GAME_LINE =
    1024; // Max number of moves in the (game + search) line that we can store

/// Piece identifiers, 4 bits each.
/// Usefull bitwise properties of this numbering scheme:
/// white = 0..., black = 1..., sliding = .1.., nonsliding = .0..
/// rank/file sliding pieces = .11., diagonally sliding pieces = .1.1
/// pawns and kings (without color bits), are < 3
/// major pieces (without color bits set), are > 5
/// minor and major pieces (without color bits set), are > 2
const int EMPTY = 0; // 0000
const int WHITE_PAWN = 1; // 0001
const int WHITE_KING = 2; // 0010
const int WHITE_KNIGHT = 3; // 0011
const int WHITE_BISHOP = 5; // 0101
const int WHITE_ROOK = 6; // 0110
const int WHITE_QUEEN = 7; // 0111
const int BLACK_PAWN = 9; // 1001
const int BLACK_KING = 10; // 1010
const int BLACK_KNIGHT = 11; // 1011
const int BLACK_BISHOP = 13; // 1101
const int BLACK_ROOK = 14; // 1110
const int BLACK_QUEEN = 15; // 1111

/// Identifier of next move:
const int WHITE_MOVE = 0;
const int BLACK_MOVE = 1;

/// Castling constants
const int CANCASTLEOO = 1; // King-side castling
const int CANCASTLEOOO = 2; // Queen-side castling

/// Square definitions (from kennyGlobals.h)
const int A8 = 56, B8 = 57, C8 = 58, D8 = 59;
const int E8 = 60, F8 = 61, G8 = 62, H8 = 63;
const int A7 = 48, B7 = 49, C7 = 50, D7 = 51;
const int E7 = 52, F7 = 53, G7 = 54, H7 = 55;
const int A6 = 40, B6 = 41, C6 = 42, D6 = 43;
const int E6 = 44, F6 = 45, G6 = 46, H6 = 47;
const int A5 = 32, B5 = 33, C5 = 34, D5 = 35;
const int E5 = 36, F5 = 37, G5 = 38, H5 = 39;
const int A4 = 24, B4 = 25, C4 = 26, D4 = 27;
const int E4 = 28, F4 = 29, G4 = 30, H4 = 31;
const int A3 = 16, B3 = 17, C3 = 18, D3 = 19;
const int E3 = 20, F3 = 21, G3 = 22, H3 = 23;
const int A2 = 8, B2 = 9, C2 = 10, D2 = 11;
const int E2 = 12, F2 = 13, G2 = 14, H2 = 15;
const int A1 = 0, B1 = 1, C1 = 2, D1 = 3;
const int E1 = 4, F1 = 5, G1 = 6, H1 = 7;

/// Piece names and characters for display
const List<String> PIECENAMES = [
  "  ",
  "P ",
  "K ",
  "N ",
  "  ",
  "B ",
  "R ",
  "Q ",
  "  ",
  "P*",
  "K*",
  "N*",
  "  ",
  "B*",
  "R*",
  "Q*",
];
const List<String> PIECECHARS = [
  " ",
  " ",
  "K",
  "N",
  " ",
  "B",
  "R",
  "Q",
  " ",
  " ",
  "K",
  "N",
  " ",
  "B",
  "R",
  "Q",
];

/// Value of material, in centipawns (from KENNY_CUSTOM_VALUES or default)
const int PAWN_VALUE = 70; // Using KENNY_CUSTOM_VALUES
const int KNIGHT_VALUE = 270;
const int BISHOP_VALUE = 280;
const int ROOK_VALUE = 470;
const int QUEEN_VALUE = 900;
const int KING_VALUE = 20000;
const int CHECK_MATE = KING_VALUE;

/// Search parameters
const int LARGE_NUMBER = KING_VALUE + 1;
const int CHECKMATESCORE = KING_VALUE;
const int STALEMATESCORE = 0;
const int DRAWSCORE = 0;

/// Quiescence and SEE (Static Exchange Evaluation) parameters
const int OFFSET = 128;
const int MINCAPTVAL = 1;
const int WEST = -1;
const int NORTHWEST = 7;
const int NORTH = 8;
const int NORTHEAST = 9;
const int EAST = 1;
const int SOUTHEAST = -7;
const int SOUTH = -8;
const int SOUTHWEST = -9;

/// Nullmove parameters
const int NULLMOVE_REDUCTION = 4;
const int NULLMOVE_LIMIT = KNIGHT_VALUE - 1;

/// Peek interval in searched node units
const int UPDATEINTERVAL = 100000;

/// Don't start a new iteration if STOPFRAC fraction of our max search time has passed:
const double STOPFRAC = 0.6;

/// Winboard constants & variables:
bool XB_MODE = false;
bool XB_PONDER = false;
bool XB_POST = false;
bool XB_DO_PENDING = false;
bool XB_NO_TIME_LIMIT = false;
const int XB_NONE = 2;
const int XB_ANALYZE = 3;
int XB_COMPUTER_SIDE = XB_NONE; // Using int for unsigned char
int XB_MIN = 0;
int XB_SEC = 0;
int XB_MPS = 0;
int XB_INC = 0;
int XB_OTIM = 0;
int XB_CTIM = 0;

/// Evaluation scores (from KENNY_CUSTOM_POSVALS or default)
const int PENALTY_DOUBLED_PAWN = 12; // Using KENNY_CUSTOM_POSVALS
const int PENALTY_ISOLATED_PAWN = 15;
const int PENALTY_BACKWARD_PAWN = 8;
const int BONUS_PASSED_PAWN = 20;
const int BONUS_BISHOP_PAIR = 50;
const int BONUS_ROOK_BEHIND_PASSED_PAWN = 12;
const int BONUS_ROOK_ON_OPEN_FILE = 20;
const int BONUS_TWO_ROOKS_ON_OPEN_FILE = 60;
const int BONUS_PAWN_SHIELD_STRONG = 18;
const int BONUS_PAWN_SHIELD_WEAK = 8;
const int MOBILITY_BONUS = 12;

/// Position values (from KENNY_CUSTOM_PSTABLES or default)
/// These arrays will need to be mirrored for black pieces during initialization.
const List<int> PAWNPOS_W = [
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  70,
  80,
  80,
  80,
  80,
  80,
  80,
  70,
  30,
  40,
  50,
  50,
  50,
  50,
  40,
  30,
  25,
  35,
  40,
  55,
  55,
  40,
  35,
  25,
  20,
  30,
  30,
  40,
  50,
  30,
  30,
  20,
  25,
  15,
  10,
  -20,
  -20,
  10,
  15,
  25,
  15,
  30,
  30,
  -60,
  -60,
  30,
  30,
  15,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
];

const List<int> KNIGHTPOS_W = [
  -32,
  -20,
  -10,
  -10,
  -10,
  -10,
  -20,
  -32,
  -20,
  -5,
  20,
  15,
  15,
  20,
  -5,
  -20,
  -10,
  24,
  40,
  44,
  44,
  40,
  24,
  -10,
  -10,
  28,
  44,
  48,
  48,
  44,
  28,
  -10,
  -10,
  30,
  44,
  48,
  48,
  44,
  30,
  -10,
  -10,
  28,
  48,
  44,
  44,
  48,
  28,
  -10,
  -10,
  0,
  24,
  28,
  28,
  24,
  0,
  -10,
  -32,
  -20,
  -10,
  -10,
  -10,
  -10,
  -20,
  -32,
];

const List<int> BISHOPPOS_W = [
  -8,
  0,
  0,
  0,
  0,
  0,
  0,
  -8,
  0,
  16,
  16,
  16,
  16,
  16,
  16,
  0,
  0,
  16,
  28,
  32,
  32,
  28,
  16,
  0,
  0,
  20,
  28,
  40,
  40,
  28,
  20,
  0,
  0,
  16,
  32,
  40,
  40,
  32,
  16,
  0,
  0,
  24,
  32,
  32,
  32,
  32,
  24,
  0,
  0,
  20,
  16,
  16,
  16,
  16,
  20,
  0,
  -8,
  0,
  -16,
  0,
  0,
  -16,
  0,
  -8,
];

const List<int> ROOKPOS_W = [
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  5,
  10,
  10,
  10,
  10,
  10,
  10,
  5,
  -5,
  0,
  0,
  0,
  0,
  0,
  0,
  -5,
  -5,
  0,
  0,
  0,
  0,
  0,
  0,
  -5,
  -5,
  0,
  0,
  0,
  0,
  0,
  0,
  -5,
  -5,
  0,
  0,
  0,
  0,
  0,
  0,
  -5,
  -5,
  0,
  0,
  0,
  0,
  0,
  0,
  -5,
  0,
  0,
  0,
  5,
  5,
  0,
  0,
  0,
];

const List<int> QUEENPOS_W = [
  -6,
  0,
  0,
  3,
  3,
  0,
  0,
  -6,
  0,
  12,
  12,
  12,
  12,
  12,
  12,
  0,
  0,
  12,
  21,
  21,
  21,
  21,
  12,
  0,
  3,
  12,
  21,
  27,
  27,
  21,
  12,
  3,
  3,
  12,
  21,
  27,
  27,
  21,
  12,
  3,
  0,
  15,
  21,
  21,
  21,
  21,
  12,
  0,
  0,
  12,
  15,
  12,
  12,
  12,
  12,
  0,
  -6,
  0,
  0,
  3,
  3,
  0,
  0,
  -6,
];

const List<int> KINGPOS_W = [
  -30,
  -40,
  -40,
  -50,
  -50,
  -40,
  -40,
  -30,
  -30,
  -40,
  -40,
  -50,
  -50,
  -40,
  -40,
  -30,
  -30,
  -40,
  -40,
  -50,
  -50,
  -40,
  -40,
  -30,
  -30,
  -40,
  -40,
  -50,
  -50,
  -40,
  -40,
  -30,
  -20,
  -30,
  -30,
  -40,
  -40,
  -30,
  -30,
  -20,
  -10,
  -20,
  -20,
  -20,
  -20,
  -20,
  -20,
  -10,
  20,
  20,
  0,
  0,
  0,
  0,
  20,
  20,
  30,
  40,
  20,
  0,
  0,
  10,
  40,
  20,
];

const List<int> KINGPOS_ENDGAME_W = [
  0,
  10,
  20,
  30,
  30,
  20,
  10,
  0,
  10,
  20,
  30,
  40,
  40,
  30,
  20,
  10,
  20,
  30,
  40,
  50,
  50,
  40,
  30,
  20,
  30,
  40,
  50,
  60,
  60,
  50,
  40,
  30,
  30,
  40,
  50,
  60,
  60,
  50,
  40,
  30,
  20,
  30,
  40,
  50,
  50,
  40,
  30,
  20,
  10,
  20,
  30,
  40,
  40,
  30,
  20,
  10,
  0,
  10,
  20,
  30,
  30,
  20,
  10,
  0,
];

const List<int> MIRROR = [
  56,
  57,
  58,
  59,
  60,
  61,
  62,
  63,
  48,
  49,
  50,
  51,
  52,
  53,
  54,
  55,
  40,
  41,
  42,
  43,
  44,
  45,
  46,
  47,
  32,
  33,
  34,
  35,
  36,
  37,
  38,
  39,
  24,
  25,
  26,
  27,
  28,
  29,
  30,
  31,
  16,
  17,
  18,
  19,
  20,
  21,
  22,
  23,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
];

const List<int> PAWN_OWN_DISTANCE = [0, 8, 4, 2, 0, 0, 0, 0];
const List<int> PAWN_OPPONENT_DISTANCE = [0, 2, 1, 0, 0, 0, 0, 0];
const List<int> KNIGHT_DISTANCE = [0, 4, 4, 0, 0, 0, 0, 0];
const List<int> BISHOP_DISTANCE = [0, 5, 4, 3, 2, 1, 0, 0];
const List<int> ROOK_DISTANCE = [0, 7, 5, 4, 3, 0, 0, 0];
const List<int> QUEEN_DISTANCE = [0, 10, 8, 5, 4, 0, 0, 0];

/// Global variables that need to be initialized at runtime or are mutable
/// In Dart, these will be non-const global variables or part of a singleton/state management.
/// For now, declaring them as `late` or with initial values.
late List<BitMap> BITSET;
late List<List<int>> BOARDINDEX; // index 0 is not used, only 1..8.
late List<int> PIECEVALUES;
late List<int> MS1BTABLE;

late List<BitMap> WHITE_PAWN_ATTACKS;
late List<BitMap> WHITE_PAWN_MOVES;
late List<BitMap> WHITE_PAWN_DOUBLE_MOVES;
late List<BitMap> BLACK_PAWN_ATTACKS;
late List<BitMap> BLACK_PAWN_MOVES;
late List<BitMap> BLACK_PAWN_DOUBLE_MOVES;
late List<BitMap> KNIGHT_ATTACKS;
late List<BitMap> KING_ATTACKS;
late List<List<BitMap>> RANK_ATTACKS;
late List<List<BitMap>> FILE_ATTACKS;
late List<List<BitMap>> DIAGA8H1_ATTACKS;
late List<List<BitMap>> DIAGA1H8_ATTACKS;

const List<int> RANKSHIFT = [
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  9,
  9,
  9,
  9,
  9,
  9,
  9,
  9,
  17,
  17,
  17,
  17,
  17,
  17,
  17,
  17,
  25,
  25,
  25,
  25,
  25,
  25,
  25,
  25,
  33,
  33,
  33,
  33,
  33,
  33,
  33,
  33,
  41,
  41,
  41,
  41,
  41,
  41,
  41,
  41,
  49,
  49,
  49,
  49,
  49,
  49,
  49,
  49,
  57,
  57,
  57,
  57,
  57,
  57,
  57,
  57,
];

const List<BitMap> FILEMAGICS = [
  0x8040201008040200,
  0x4020100804020100,
  0x2010080402010080,
  0x1008040201008040,
  0x0804020100804020,
  0x0402010080402010,
  0x0201008040201008,
  0x0100804020100804,
];

const List<BitMap> DIAGA8H1MAGICS = [
  0x0,
  0x0,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x0080808080808080,
  0x0040404040404040,
  0x0020202020202020,
  0x0010101010101010,
  0x0008080808080808,
  0x0,
  0x0,
];

const List<BitMap> DIAGA1H8MAGICS = [
  0x0,
  0x0,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x0101010101010100,
  0x8080808080808000,
  0x4040404040400000,
  0x2020202020000000,
  0x1010101000000000,
  0x0808080000000000,
  0x0,
  0x0,
];

late List<BitMap> RANKMASK;
late List<BitMap> FILEMASK;
late List<BitMap> FILEMAGIC;
late List<BitMap> DIAGA8H1MASK;
late List<BitMap> DIAGA8H1MAGIC;
late List<BitMap> DIAGA1H8MASK;
late List<BitMap> DIAGA1H8MAGIC;

late List<List<int>> GEN_SLIDING_ATTACKS; // unsigned char becomes int

late BitMap maskEG0;
late BitMap maskFG0;
late BitMap maskBD0;
late BitMap maskCE0;
late BitMap maskEG1;
late BitMap maskFG1;
late BitMap maskBD1;
late BitMap maskCE1;

late int WHITE_OOO_CASTL;
late int BLACK_OOO_CASTL;
late int WHITE_OO_CASTL;
late int BLACK_OO_CASTL;

late int ICAPT;
late int IEP;
late int IPROM;
late int ICASTLOO;
late int ICASTLOOO;
late int ICHECK;

late List<int> PAWNPOS_B;
late List<int> KNIGHTPOS_B;
late List<int> BISHOPPOS_B;
late List<int> ROOKPOS_B;
late List<int> QUEENPOS_B;
late List<int> KINGPOS_B;
late List<int> KINGPOS_ENDGAME_B;

late List<BitMap> PASSED_WHITE;
late List<BitMap> PASSED_BLACK;
late List<BitMap> ISOLATED_WHITE;
late List<BitMap> ISOLATED_BLACK;
late List<BitMap> BACKWARD_WHITE;
late List<BitMap> BACKWARD_BLACK;
late List<BitMap> KINGSHIELD_STRONG_W;
late List<BitMap> KINGSHIELD_STRONG_B;
late List<BitMap> KINGSHIELD_WEAK_W;
late List<BitMap> KINGSHIELD_WEAK_B;
late BitMap WHITE_SQUARES;
late BitMap BLACK_SQUARES;

late List<List<int>> DISTANCE;

late List<BitMap> RAY_W;
late List<BitMap> RAY_NW;
late List<BitMap> RAY_N;
late List<BitMap> RAY_NE;
late List<BitMap> RAY_E;
late List<BitMap> RAY_SE;
late List<BitMap> RAY_S;
late List<BitMap> RAY_SW;
late List<List<int>> HEADINGS;



// Global variables for Winboard communication (from kennyExtGlobals.h)
// These are already declared above as `bool` or `int` with initial values.
// char CMD_BUFF[MAX_CMD_BUFF]; // Handled by input stream
// int CMD_BUFF_COUNT = 0;
// char INIFILE[80]; // Handled by file operations
// char PATHNAME[80]; // Handled by file operations

// Keep track of stdout (writing to a file or to the console):
// In Dart, console output is typically handled by `print` and file I/O by `dart:io`.
// This variable might not have a direct Dart equivalent if the output redirection
// is handled differently. For now, it's omitted.
// int TOCONSOLE;
