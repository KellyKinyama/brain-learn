/// kenny_uci.dart
///
/// This file implements the Universal Chess Interface (UCI) protocol for the
/// Kenny chess engine, translating the functionality found in Stockfish's
/// uci.cpp and uci.h files into Dart.
///
/// It handles parsing UCI commands from stdin and sending appropriate
/// responses to stdout.

import 'dart:io';
import 'dart:async';
import 'package:collection/collection.dart'; // For list comparison

// Import your actual engine files
import 'board.dart';
import 'defs.dart'; // For SQUARENAME, PIECECHARS, NOMOVE, etc.
import 'move.dart';
import 'display_move.dart'; // For formatMove (and potentially toSan)
import 'make_move.dart'; // For makeMove, unmakeMove
import 'move_gen.dart'; // For movegen, isOwnKingAttacked
import 'perft.dart'; // For perft function
import 'read_fen.dart'; // For readFenString

/// The main Engine class that encapsulates the chess logic and search.
/// This replaces the previous MockEngine and integrates with your actual Board.
class Engine {
  final Map<String, dynamic> _options;
  bool _searchRunning = false;
  Completer<void>? _searchCompleter;

  Engine(this._options); // Constructor now takes options

  /// Sets the board position from FEN and applies moves.
  /// Uses your actual `board` global instance and `makeMove` function.
  void setPosition(String fen, List<String> moves) {
    readFenString(fen); // Use your actual FEN reader
    for (var moveStr in moves) {
      final parsedMove = _parseMove(moveStr); // Use the UCI engine's move parser
      if (parsedMove != NOMOVE) {
        makeMove(parsedMove); // Use your actual makeMove function
      } else {
        stdout.writeln('info string Warning: Could not parse or make move: $moveStr');
      }
    }
  }

  /// Starts the search based on given limits.
  /// This will call your `board.think()` method for actual search.
  Future<void> go(Map<String, dynamic> limits) async {
    if (_searchRunning) {
      stdout.writeln('info string Search already running. Stopping current search.');
      stop();
      await _searchCompleter?.future; // Wait for previous search to stop
    }

    _searchRunning = true;
    _searchCompleter = Completer<void>();

    stdout.writeln('info string Starting search with limits: $limits');

    // Here, you would integrate your actual search logic.
    // For now, we'll call a simplified mock search or your board.think().
    // If your board.think() is blocking, you might need to run it in an Isolate.

    // Example of calling your engine's think method:
    // board.maxTime = limits['movetime'] ?? board.maxTime; // Set max time if provided
    // board.searchDepth = limits['depth'] ?? board.searchDepth; // Set depth if provided
    // board.ponder = limits['ponder'] ?? false; // Set ponder mode

    // In a real scenario, board.think() would be a long-running operation
    // that reports progress via callbacks or streams, which you'd then
    // translate into UCI 'info' strings.
    // For this integration, we'll simulate it with a Future.delayed
    // and then call board.think() to get a move.

    // Simulate search progress (replace with actual search progress reporting)
    int depth = 0;
    int nodes = 0;
    int score = 0; // Centipawns
    String pv = '';

    // This loop simulates the iterative deepening and info output
    while (_searchRunning && depth < (limits['depth'] ?? 5)) {
      await Future.delayed(Duration(milliseconds: 100)); // Simulate work
      depth++;
      nodes += 5000; // Mock nodes
      score = board.eval(); // Get current board evaluation (mock or actual)

      // In a real engine, you'd get the PV from your search algorithm.
      // For now, let's use a placeholder or derive from board.lastPV.
      if (board.lastPVLength > 0) {
        pv = board.lastPV.sublist(0, board.lastPVLength).map((m) => formatMove(m)).join(' ');
      } else {
        pv = 'e2e4 e7e5'; // Default mock PV
      }


      stdout.writeln('info depth $depth seldepth ${depth + 1} score cp $score '
          'nodes $nodes nps ${nodes * 10} hashfull 500 tbhits 0 time ${depth * 100} pv $pv');
    }

    if (_searchRunning) {
      // Call your actual board.think() to get the best move
      final bestMove = board.think(); // This should run your search and return the best move
      final ponderMove = NOMOVE; // Your engine needs to determine the ponder move

      _sendBestMove(bestMove, ponderMove);
    } else {
      stdout.writeln('info string Search was stopped.');
    }

    _searchRunning = false;
    _searchCompleter?.complete();
  }

  /// Stops the current search.
  void stop() {
    _searchRunning = false;
    // In your actual engine, you would set a flag that the search loop
    // checks to terminate gracefully.
    stdout.writeln('info string Stop signal received.');
  }

  /// Clears search-related data (e.g., hash tables).
  void searchClear() {
    // Implement clearing of hash tables, transposition tables, etc.
    stdout.writeln('info string Search data cleared.');
  }

  /// Sends the bestmove and optional ponder move.
  void _sendBestMove(Move bestMove, [Move? ponderMove]) {
    String bestMoveStr = formatMove(bestMove);
    String ponderMoveStr = ponderMove != NOMOVE ? ' ponder ${formatMove(ponderMove)}' : '';
    stdout.writeln('bestmove $bestMoveStr$ponderMoveStr');
  }

  /// Converts a UCI move string (e.g., "e2e4", "g1f3", "e7e8q") to a Move object.
  /// This implementation iterates through legal moves to find a match.
  Move _parseMove(String moveStr) {
    // Generate legal moves for the current board position
    final legalMoves = board.generateLegalMoves();

    // Iterate through legal moves and find a match based on UCI string format
    for (var move in legalMoves) {
      if (formatMove(move).toLowerCase() == moveStr.toLowerCase()) {
        return move;
      }
    }
    return NOMOVE; // Return NOMOVE if no legal move matches
  }
}


/// The main UCI Engine class.
class UCIEngine {
  final Engine _engine;
  final Map<String, dynamic> _options = {
    'Hash': 16, // Default hash size in MB
    'Threads': 1, // Default number of threads
    'Ponder': false, // Default ponder mode
    'UCI_ShowWDL': false, // Default for showing WDL
  };

  UCIEngine() : _engine = Engine({}) {
    // Initialize the global board instance (if not already done in main.dart)
    // Assuming 'board' is initialized globally in main.dart or similar.
    // If not, you might need: board = Board(); board.init(); here.
    // Pass the options map to the engine
    _engine._options.addAll(_options);
  }

  /// The main loop for processing UCI commands.
  Future<void> loop() async {
    stdin.transform(SystemEncoding().decoder).transform(const LineSplitter()).listen((line) async {
      line = line.trim();
      if (line.isEmpty) return;

      final parts = line.split(' ');
      final command = parts[0];

      switch (command) {
        case 'uci':
          _handleUci();
          break;
        case 'isready':
          _handleIsReady();
          break;
        case 'setoption':
          _handleSetOption(line);
          break;
        case 'ucinewgame':
          _handleUciNewGame();
          break;
        case 'position':
          await _handlePosition(line);
          break;
        case 'go':
          await _handleGo(line);
          break;
        case 'stop':
          _handleStop();
          break;
        case 'ponderhit':
          _handlePonderHit();
          break;
        case 'quit':
          stdout.writeln('info string Quitting engine.');
          exit(0);
        default:
          stdout.writeln('info string Unknown command: $line');
      }
    });
  }

  /// Handles the 'uci' command.
  void _handleUci() {
    stdout.writeln('id name Kenny v0.1');
    stdout.writeln('id author Kenshin Himura');

    // Output engine options
    _options.forEach((key, value) {
      String type;
      String defaultValue = value.toString();
      String extra = '';

      if (value is int) {
        type = 'spin';
        // Add min/max for spin options if applicable
        if (key == 'Hash') extra = 'min 1 max 2048'; // Example limits
        if (key == 'Threads') extra = 'min 1 max 128'; // Example limits
      } else if (value is bool) {
        type = 'check';
      } else {
        type = 'string'; // Fallback for other types
      }
      stdout.writeln('option name $key type $type default $defaultValue $extra'.trim());
    });

    stdout.writeln('uciok');
  }

  /// Handles the 'isready' command.
  void _handleIsReady() {
    stdout.writeln('readyok');
  }

  /// Handles the 'setoption' command.
  void _handleSetOption(String line) {
    // Example: setoption name Hash value 128
    final parts = line.split(' ');
    if (parts.length < 5 || parts[1] != 'name' || parts[3] != 'value') {
      stdout.writeln('info string Malformed setoption command: $line');
      return;
    }

    final optionName = parts[2];
    final optionValue = parts.sublist(4).join(' '); // Get the full value string

    if (_options.containsKey(optionName)) {
      try {
        dynamic parsedValue;
        if (_options[optionName] is int) {
          parsedValue = int.parse(optionValue);
        } else if (_options[optionName] is bool) {
          parsedValue = optionValue.toLowerCase() == 'true';
        } else {
          parsedValue = optionValue;
        }
        _options[optionName] = parsedValue;
        _engine._options[optionName] = parsedValue; // Update engine's options
        stdout.writeln('info string Set option $optionName to $parsedValue');
      } catch (e) {
        stdout.writeln('info string Error parsing value for option $optionName: $e');
      }
    } else {
      stdout.writeln('info string Unknown option: $optionName');
    }
  }

  /// Handles the 'ucinewgame' command.
  void _handleUciNewGame() {
    _engine.searchClear();
    // Reset board to initial state
    board.init(); // Use your actual board.init()
    stdout.writeln('info string New game started.');
  }

  /// Handles the 'position' command.
  Future<void> _handlePosition(String line) async {
    final parts = line.split(' ');
    String fen = '';
    List<String> moves = [];

    int fenStartIndex = -1;
    int movesStartIndex = -1;

    if (parts[1] == 'startpos') {
      fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      fenStartIndex = 1; // 'startpos' itself
    } else if (parts[1] == 'fen') {
      fenStartIndex = line.indexOf('fen') + 4;
      movesStartIndex = line.indexOf('moves', fenStartIndex);

      if (movesStartIndex != -1) {
        fen = line.substring(fenStartIndex, movesStartIndex).trim();
      } else {
        fen = line.substring(fenStartIndex).trim();
      }
    } else {
      stdout.writeln('info string Invalid position command: $line');
      return;
    }

    if (movesStartIndex != -1) {
      final movesString = line.substring(movesStartIndex + 6).trim();
      moves = movesString.split(' ');
    }

    _engine.setPosition(fen, moves);
    stdout.writeln('info string Position set.');
  }

  /// Handles the 'go' command.
  Future<void> _handleGo(String line) async {
    final limits = _parseLimits(line);
    await _engine.go(limits);
  }

  /// Handles the 'stop' command.
  void _handleStop() {
    _engine.stop();
  }

  /// Handles the 'ponderhit' command.
  void _handlePonderHit() {
    stdout.writeln('info string Ponderhit received.');
    // In a real engine, this would switch from ponder mode to normal search.
  }

  /// Parses search limits from the 'go' command line.
  Map<String, dynamic> _parseLimits(String line) {
    final limits = <String, dynamic>{};
    final parts = line.split(' ');

    for (int i = 1; i < parts.length; i++) {
      final token = parts[i];
      switch (token) {
        case 'searchmoves':
          limits['searchmoves'] = <String>[];
          while (++i < parts.length && parts[i] != 'wtime' && parts[i] != 'btime' &&
                 parts[i] != 'winc' && parts[i] != 'binc' && parts[i] != 'movestogo' &&
                 parts[i] != 'depth' && parts[i] != 'nodes' && parts[i] != 'movetime' &&
                 parts[i] != 'mate' && parts[i] != 'perft' && parts[i] != 'infinite' &&
                 parts[i] != 'ponder') {
            (limits['searchmoves'] as List<String>).add(parts[i]);
          }
          i--; // Adjust index as loop increments it one too many
          break;
        case 'wtime':
          limits['wtime'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'btime':
          limits['btime'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'winc':
          limits['winc'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'binc':
          limits['binc'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'movestogo':
          limits['movestogo'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'depth':
          limits['depth'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'nodes':
          limits['nodes'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'movetime':
          limits['movetime'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'mate':
          limits['mate'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'perft':
          limits['perft'] = int.tryParse(parts[++i]) ?? 0;
          break;
        case 'infinite':
          limits['infinite'] = true;
          break;
        case 'ponder':
          limits['ponder'] = true;
          break;
      }
    }
    return limits;
  }
}

/// Main function to start the UCI engine.
void main() {
  // Initialize your global board instance here, as done in your main.dart
  dataInit(); // Assuming this initializes global data like KEY, PIECEVALUES etc.
  board = Board();
  board.init();

  final engine = UCIEngine();
  stdout.writeln('Kenny chess engine started. Type "uci" to begin.');
  engine.loop();
}
