// /// kenny_board.dart
// ///
// /// This file defines the `Board` class, representing the chess board state
// /// and containing methods for game logic, move generation, and search.
// /// It translates the C++ `Board` struct from kennyBoard.h.

// import 'defs.dart';
// import 'move.dart';
// import 'game_line.dart';
// import 'hash.dart';
// import 'timer.dart';
// import 'utils.dart';
// import 'bit_ops.dart';

// // Global instance of the Board (from kennyGlobals.h)
// late Board board;

// class Board {
//   // Bitboards for piece positions
//   BitMap whiteKing,
//       whiteQueens,
//       whiteRooks,
//       whiteBishops,
//       whiteKnights,
//       whitePawns;
//   BitMap blackKing,
//       blackQueens,
//       blackRooks,
//       blackBishops,
//       blackKnights,
//       blackPawns;
//   BitMap whitePieces, blackPieces, occupiedSquares;

//   // Game state variables
//   int nextMove; // WHITE_MOVE or BLACK_MOVE
//   int castleWhite; // White's castle status, CANCASTLEOO = 1, CANCASTLEOOO = 2
//   int castleBlack; // Black's castle status, CANCASTLEOO = 1, CANCASTLEOOO = 2
//   int epSquare; // En-passant target square after double pawn move
//   int fiftyMove; // Moves since the last pawn move or capture
//   U64 hashkey; // Random 'almost' unique signature for current board position.

//   // Additional variables
//   List<int>
//   square; // incrementally updated, what kind of piece is on a particular square.
//   int
//   Material; // incrementally updated, total material balance on board, in centipawns.
//   int totalWhitePawns; // sum of P material value for white (in centipawns)
//   int totalBlackPawns; // sum of P material value for black (in centipawns)
//   int
//   totalWhitePieces; // sum of Q+R+B+N material value for white (in centipawns)
//   int
//   totalBlackPieces; // sum of Q+R+B+N material value for black (in centipawns)

//   bool viewRotated; // only used for displaying the board. TRUE or FALSE.

//   // Storing moves:
//   List<Move>
//   moveBuffer; // all generated moves of the current search tree are stored in this array.
//   List<int>
//   moveBufLen; // this arrays keeps track of which moves belong to which ply
//   int endOfGame; // index for board.gameLine
//   int endOfSearch; // index for board.gameLine
//   List<GameLineRecord> gameLine;

//   // Search variables:
//   List<int> triangularLength;
//   List<List<Move>> triangularArray;
//   Timer timer;
//   U64 msStart, msStop;
//   int searchDepth;
//   int lastPVLength;
//   List<Move> lastPV;
//   List<List<int>> whiteHeuristics;
//   List<List<int>> blackHeuristics;
//   bool followpv;
//   bool allownull;
//   U64 inodes;
//   U64 countdown;
//   U64 maxTime;
//   bool timedout;
//   bool ponder;

//   /// Constructor to initialize the Board with default values.
//   Board()
//     : whiteKing = 0,
//       whiteQueens = 0,
//       whiteRooks = 0,
//       whiteBishops = 0,
//       whiteKnights = 0,
//       whitePawns = 0,
//       blackKing = 0,
//       blackQueens = 0,
//       blackRooks = 0,
//       blackBishops = 0,
//       blackKnights = 0,
//       blackPawns = 0,
//       whitePieces = 0,
//       blackPieces = 0,
//       occupiedSquares = 0,
//       nextMove = WHITE_MOVE,
//       castleWhite = 0,
//       castleBlack = 0,
//       epSquare = 0,
//       fiftyMove = 0,
//       hashkey = 0,
//       square = List.filled(64, EMPTY),
//       Material = 0,
//       totalWhitePawns = 0,
//       totalBlackPawns = 0,
//       totalWhitePieces = 0,
//       totalBlackPieces = 0,
//       viewRotated = false,
//       moveBuffer = List.generate(MAX_MOV_BUFF, (index) => Move()),
//       moveBufLen = List.filled(MAX_PLY, 0),
//       endOfGame = 0,
//       endOfSearch = 0,
//       gameLine = List.generate(
//         MAX_GAME_LINE,
//         (index) => GameLineRecord.empty(),
//       ),
//       triangularLength = List.filled(MAX_PLY, 0),
//       triangularArray = List.generate(
//         MAX_PLY,
//         (_) => List.generate(MAX_PLY, (__) => Move()),
//       ),
//       timer = Timer(),
//       msStart = 0,
//       msStop = 0,
//       searchDepth = 16, // Default from C++ kennyData.cpp
//       lastPVLength = 0,
//       lastPV = List.generate(MAX_PLY, (index) => Move()),
//       whiteHeuristics = List.generate(64, (_) => List.filled(64, 0)),
//       blackHeuristics = List.generate(64, (_) => List.filled(64, 0)),
//       followpv = false,
//       allownull = false,
//       inodes = 0,
//       countdown = 0,
//       maxTime = 2000, // Default from C++ kennyData.cpp (milliseconds)
//       timedout = false,
//       ponder = false;

//   /// Initializes the board to the starting position.
//   /// Translates Board::init() from kennyBoard.cpp.
//   void init() {
//     viewRotated = false;

//     for (int i = 0; i < 64; i++) {
//       square[i] = EMPTY;
//     }

//     // Set up initial piece positions
//     square[E1] = WHITE_KING;
//     square[D1] = WHITE_QUEEN;
//     square[A1] = WHITE_ROOK;
//     square[H1] = WHITE_ROOK;
//     square[B1] = WHITE_KNIGHT;
//     square[G1] = WHITE_KNIGHT;
//     square[C1] = WHITE_BISHOP;
//     square[F1] = WHITE_BISHOP;
//     for (int i = A2; i <= H2; i++) square[i] = WHITE_PAWN;

//     square[E8] = BLACK_KING;
//     square[D8] = BLACK_QUEEN;
//     square[A8] = BLACK_ROOK;
//     square[H8] = BLACK_ROOK;
//     square[B8] = BLACK_KNIGHT;
//     square[G8] = BLACK_KNIGHT;
//     square[C8] = BLACK_BISHOP;
//     square[F8] = BLACK_BISHOP;
//     for (int i = A7; i <= H7; i++) square[i] = BLACK_PAWN;

//     initFromSquares(
//       square,
//       WHITE_MOVE,
//       0,
//       CANCASTLEOO | CANCASTLEOOO,
//       CANCASTLEOO | CANCASTLEOOO,
//       0,
//     );
//   }

//   /// Initializes the board from a given square array and game state.
//   /// Translates Board::initFromSquares() from kennyBoard.cpp.
//   void initFromSquares(
//     List<int> inputSquares,
//     int next,
//     int fiftyM,
//     int castleW,
//     int castleB,
//     int epSq,
//   ) {
//     // Clear all bitboards
//     whiteKing = 0;
//     whiteQueens = 0;
//     whiteRooks = 0;
//     whiteBishops = 0;
//     whiteKnights = 0;
//     whitePawns = 0;
//     blackKing = 0;
//     blackQueens = 0;
//     blackRooks = 0;
//     blackBishops = 0;
//     blackKnights = 0;
//     blackPawns = 0;

//     // Clear material counts
//     totalWhitePawns = 0;
//     totalBlackPawns = 0;
//     totalWhitePieces = 0;
//     totalBlackPieces = 0;

//     // Populate bitboards and calculate material
//     for (int i = 0; i < 64; i++) {
//       square[i] = inputSquares[i]; // Copy input squares
//       switch (inputSquares[i]) {
//         case WHITE_PAWN:
//           whitePawns |= BITSET[i];
//           totalWhitePawns += PAWN_VALUE;
//           break;
//         case WHITE_KNIGHT:
//           whiteKnights |= BITSET[i];
//           totalWhitePieces += KNIGHT_VALUE;
//           break;
//         case WHITE_BISHOP:
//           whiteBishops |= BITSET[i];
//           totalWhitePieces += BISHOP_VALUE;
//           break;
//         case WHITE_ROOK:
//           whiteRooks |= BITSET[i];
//           totalWhitePieces += ROOK_VALUE;
//           break;
//         case WHITE_QUEEN:
//           whiteQueens |= BITSET[i];
//           totalWhitePieces += QUEEN_VALUE;
//           break;
//         case WHITE_KING:
//           whiteKing |= BITSET[i];
//           break;
//         case BLACK_PAWN:
//           blackPawns |= BITSET[i];
//           totalBlackPawns += PAWN_VALUE;
//           break;
//         case BLACK_KNIGHT:
//           blackKnights |= BITSET[i];
//           totalBlackPieces += KNIGHT_VALUE;
//           break;
//         case BLACK_BISHOP:
//           blackBishops |= BITSET[i];
//           totalBlackPieces += BISHOP_VALUE;
//           break;
//         case BLACK_ROOK:
//           blackRooks |= BITSET[i];
//           totalBlackPieces += ROOK_VALUE;
//           break;
//         case BLACK_QUEEN:
//           blackQueens |= BITSET[i];
//           totalBlackPieces += QUEEN_VALUE;
//           break;
//         case BLACK_KING:
//           blackKing |= BITSET[i];
//           break;
//         default:
//           break;
//       }
//     }

//     // Set the final material score (difference)
//     Material =
//         (totalWhitePawns + totalWhitePieces) -
//         (totalBlackPawns + totalBlackPieces);

//     whitePieces =
//         whitePawns |
//         whiteKnights |
//         whiteBishops |
//         whiteRooks |
//         whiteQueens |
//         whiteKing;
//     blackPieces =
//         blackPawns |
//         blackKnights |
//         blackBishops |
//         blackRooks |
//         blackQueens |
//         blackKing;
//     occupiedSquares = whitePieces | blackPieces;

//     nextMove = next;
//     fiftyMove = fiftyM;
//     castleWhite = castleW;
//     castleBlack = castleB;
//     epSquare = epSq;

//     // Calculate initial hash key
//     hashkey = 0;
//     for (int i = 0; i < 64; i++) {
//       if (square[i] != EMPTY) {
//         hashkey ^= KEY.keys[i][square[i]];
//       }
//     }
//     if (nextMove == BLACK_MOVE) hashkey ^= KEY.side;
//     if ((castleWhite & CANCASTLEOO) != 0) hashkey ^= KEY.wk;
//     if ((castleWhite & CANCASTLEOOO) != 0) hashkey ^= KEY.wq;
//     if ((castleBlack & CANCASTLEOO) != 0) hashkey ^= KEY.bk;
//     if ((castleBlack & CANCASTLEOOO) != 0) hashkey ^= KEY.bq;
//     if (epSquare != 0) hashkey ^= KEY.ep[epSquare];

//     // Reset search-related variables
//     endOfGame = 0;
//     endOfSearch = 0;
//     lastPVLength = 0;
//     inodes = 0;
//     timedout = false;
//     countdown = UPDATEINTERVAL;
//   }

//   /// Displays the board on the console.
//   /// Translates Board::display() from kennyBoard.cpp.
//   void display() {
//     // This will be a simplified text-based display.
//     // For a graphical UI, this method would be replaced by rendering logic.
//     print('\n    +---+---+---+---+---+---+---+---+');
//     for (int rank = 8; rank >= 1; rank--) {
//       String row = '    |';
//       for (int file = 1; file <= 8; file++) {
//         int sqIndex = BOARDINDEX[file][rank];
//         row += ' ${PIECENAMES[square[sqIndex]]}|';
//       }
//       print('$row $rank');
//       print('    +---+---+---+---+---+---+---+---+');
//     }
//     print('      a   b   c   d   e   f   g   h\n');

//     print('next=${nextMove == WHITE_MOVE ? 'WHITE' : 'BLACK'}');
//     print('ep=${epSquare == 0 ? '-' : SQUARENAME[epSquare]}');
//     print('fifty=$fiftyMove');
//     print('white castle=$castleWhite');
//     print('black castle=$castleBlack');
//     print('key=$hashkey');
//   }

//   /// Placeholder for eval() function.
//   /// Translates Board::eval() from kennyEval.cpp.
//   int eval() {
//     // This is a complex function in C++ and will require detailed translation.
//     // For now, return a placeholder score.
//     // The C++ eval function calculates score from White's perspective and then
//     // returns it from the perspective of the side to move.
//     int score = Material; // Start with material balance

//     // Placeholder for positional bonuses, pawn structure, king safety etc.
//     // The full implementation would involve iterating through pieces,
//     // using position tables, and evaluating various chess concepts.

//     if (nextMove == BLACK_MOVE) {
//       return -score; // Return from side to move's perspective
//     }
//     return score;
//   }

//   /// Placeholder for think() function.
//   /// Translates Board::think() from kennySearch.cpp.
//   Move think() {
//     // This is the main search function, involving iterative deepening,
//     // alpha-beta pruning, and quiescence search.
//     // It will be a complex translation. For now, return a dummy move.
//     print('Thinking...');
//     // In a real implementation, this would call `alphabeta` or `qsearch`
//     // and return the best move found.
//     return Move(moveInt: 0); // Return a default/invalid move
//   }

//   /// Placeholder for minimax() function.
//   /// Translates Board::minimax() from kennySearch.cpp (if used, often replaced by alphabeta).
//   int minimax(int ply, int depth) {
//     // Minimax is a basic search algorithm, usually optimized with Alpha-Beta.
//     // This function is likely a helper or an older version of the search.
//     // Implement if needed for specific search modes.
//     return 0;
//   }

//   /// Placeholder for alphabeta() function.
//   /// Translates Board::alphabeta() from kennySearch.cpp.
//   int alphabeta(int ply, int depth, int alpha, int beta) {
//     // This is the core alpha-beta search algorithm.
//     // Will involve move generation, make/unmake move, recursion, and evaluation.
//     return 0;
//   }

//   /// Placeholder for alphabetapvs() function (Principal Variation Search).
//   /// Translates Board::alphabetapvs() from kennySearch.cpp.
//   int alphabetapvs(int ply, int depth, int alpha, int beta) {
//     // PVS is an optimization of alpha-beta.
//     return 0;
//   }

//   /// Placeholder for qsearch() function (Quiescence Search).
//   /// Translates Board::qsearch() from kennyQSearch.cpp.
//   int qsearch(int ply, int alpha, int beta) {
//     // Quiescence search extends the search at the leaves of the main search tree
//     // to ensure that the evaluation is performed on a "quiet" position (no immediate captures/promotions).
//     return 0;
//   }

//   /// Placeholder for displaySearchStats() function.
//   /// Translates Board::displaySearchStats() from kennySearch.cpp.
//   void displaySearchStats(int mode, int depth, int score) {
//     // Displays information about the search progress.
//     print('Search Stats: Depth=$depth, Score=$score');
//   }

//   /// Placeholder for isEndOfgame() function.
//   /// Translates Board::isEndOfgame() from kennySearch.cpp.
//   bool isEndOfgame(int legalmoves, Move singlemove) {
//     // Checks if the game has ended (checkmate, stalemate, draw).
//     return false;
//   }

//   /// Placeholder for repetitionCount() function.
//   /// Translates Board::repetitionCount() from kennySearch.cpp.
//   int repetitionCount() {
//     // Checks for three-fold repetition.
//     return 0;
//   }

//   /// Placeholder for mirror() function.
//   /// Translates Board::mirror() from kennyBoard.cpp.
//   void mirror() {
//     // Mirrors the board state (useful for symmetric evaluation).
//   }

//   /// Placeholder for rememberPV() function.
//   /// Translates Board::rememberPV() from kennySearch.cpp.
//   void rememberPV() {
//     // Stores the principal variation (best line of play found so far).
//   }

//   /// Placeholder for selectmove() function.
//   /// Translates Board::selectmove() from kennySortMoves.cpp.
//   void selectmove(int ply, int i, int depth, bool followpv) {
//     // Reorders the move list for better alpha-beta pruning.
//   }

//   /// Placeholder for addCaptScore() function.
//   /// Translates Board::addCaptScore() from kennyMoveGen.cpp.
//   void addCaptScore(int ifirst, int index) {
//     // Adds a score to a capture move for move ordering.
//   }

//   /// Placeholder for SEE() function (Static Exchange Evaluation).
//   /// Translates Board::SEE() from kennySEE.cpp.
//   int SEE(Move move) {
//     // Calculates the Static Exchange Evaluation for a move.
//     return 0;
//   }

//   /// Placeholder for attacksTo() function.
//   /// Translates Board::attacksTo() from kennySEE.cpp.
//   BitMap attacksTo(int target) {
//     // Returns a bitboard of pieces attacking a target square.
//     return 0;
//   }

//   /// Placeholder for revealNextAttacker() function.
//   /// Translates Board::revealNextAttacker() from kennySEE.cpp.
//   BitMap revealNextAttacker(
//     BitMap attackers,
//     BitMap nonremoved,
//     int target,
//     int heading,
//   ) {
//     // Used in SEE to find the next attacker after a piece is removed.
//     return 0;
//   }

//   /// Placeholder for readClockAndInput() function.
//   /// Translates Board::readClockAndInput() from kennyPeek.cpp.
//   void readClockAndInput() {
//     // Checks for time limits and user input during search.
//   }
// }
