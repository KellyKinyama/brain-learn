/// kenny_commands.dart
///
/// This file translates the command processing logic from kennyCommands.cpp.
/// It handles various commands, including those for Winboard protocol.
/// Due to Dart's different I/O model and lack of direct `sscanf` equivalents,
/// string parsing will use Dart's built-in String methods and `int.tryParse`.

import 'dart:io';
import 'defs.dart';
import 'board.dart';
import 'move.dart';
import 'display_move.dart'; // For toSan
import 'read_fen.dart'; // For readFen
import 'data.dart'; // For dataInit
import 'perft.dart'; // For perft

/// Processes a single command string.
/// This function simulates the behavior of the `commands()` function in C++.
/// It interacts with the global `board` object and global Winboard flags.
Future<void> processCommand(String commandLine) async {
  List<String> parts = commandLine.trim().split(' ');
  String command = parts[0].toLowerCase();

  // Helper for parsing integers safely
  int? parseInt(String? s) => s != null ? int.tryParse(s) : null;

  // =================================================================
  // help: display help
  // =================================================================
  if (command == "help") {
    print("Kenny Chess Engine Commands:");
    print("  help                               - Display this help message.");
    print("  new                                - Start a new game.");
    print("  undo                               - Undo the last move.");
    print(
      "  go                                 - Make the engine think and play a move.",
    );
    print(
      "  perft <depth>                      - Run a performance test to specified depth.",
    );
    print("  d                                  - Display the current board.");
    print("  fen <fen_string>                   - Set board from FEN string.");
    print("  setboard <fen_string>              - Same as fen.");
    print("  sd <depth>                         - Set search depth.");
    print("  st <time_in_seconds>               - Set search time.");
    print(
      "  time <ms>                          - Set current time for engine (Winboard).",
    );
    print(
      "  otim <ms>                          - Set opponent time (Winboard).",
    );
    print("  protover <version>                 - Winboard protocol version.");
    print("  xboard                             - Enter Winboard mode.");
    print("  quit                               - Exit the program.");
    print(
      "  <move>                             - Make a move (e.g., e2e4, e7e8q).",
    );
    return;
  }

  // =================================================================
  // new: start a new game
  // =================================================================
  if (command == "new") {
    board.init(); // Reset board to initial position
    XB_COMPUTER_SIDE = XB_NONE;
    XB_POST = false;
    print("New game started. Board reset.");
    board.display();
    return;
  }

  // =================================================================
  // undo: undo the last move
  // =================================================================
  if (command == "undo") {
    if (board.endOfGame > 0) {
      // In C++, unmakeMove is called directly. We need to ensure
      // the gameLine is correctly managed.
      // This part assumes `unmakeMove` will correctly restore the previous state.
      // The C++ `undo` command logic is in `commands.cpp` and calls `unmakeMove`.
      // It also needs to handle `board.endOfGame` decrement.
      // For now, a simplified undo.
      // A proper undo would require storing the full GameLineRecord.
      // Assuming `board.gameLine` stores the history and `endOfGame` is the current index.
      if (board.endOfGame > 0) {
        board.endOfGame--;
        // Restore board state from gameLine[endOfGame]
        // This is a simplified restoration. A full `unmakeMove` is needed.
        // For now, we'll just report.
        print(
          "Undo command received. (Actual unmakeMove logic not fully translated yet)",
        );
        // TODO: Implement full unmakeMove and state restoration here or in Board class.
      } else {
        print("No moves to undo.");
      }
    } else {
      print("No moves to undo.");
    }
    return;
  }

  // =================================================================
  // go: make the engine think and play a move
  // =================================================================
  if (command == "go") {
    if (XB_MODE) {
      XB_COMPUTER_SIDE = board.nextMove;
    }
    print("Engine is thinking...");
    Move bestMove = board.think(); // This will trigger the search
    if (bestMove.moveInt != 0) {
      // Assuming think() returns the best move.
      // makeMove will update the board.
      // displayMove or toSan will format the move.
      String sanMove = '';
      if (toSan(bestMove, sanMove)) {
        print("move $sanMove"); // Winboard format
      } else {
        print("move ${bestMove.toString()}"); // Fallback
      }
      // makeMove(bestMove); // Board state update is handled within think() or after.
      // In C++, think() returns the move, then it's made.
      // Let's call makeMove here for clarity.
      // makeMove(bestMove); // This should be handled by the search process.
      // The C++ `think` function already makes the move internally.
      // We just need to display it.
    } else {
      print("No legal moves or game ended.");
    }
    return;
  }

  // =================================================================
  // perft: run a performance test
  // =================================================================
  if (command == "perft") {
    if (parts.length < 2) {
      print("Usage: perft <depth>");
      return;
    }
    int? depth = parseInt(parts[1]);
    if (depth == null || depth <= 0) {
      print("Invalid depth for perft: ${parts[1]}");
      return;
    }
    print("Running perft to depth $depth...");
    board.timer.init();
    U64 nodes = perft(0, depth); // Call the global perft function
    board.timer.stop();
    print("Nodes: $nodes");
    print("Time: ${board.timer.display()}s");
    return;
  }

  // =================================================================
  // d: display the current board
  // =================================================================
  if (command == "d") {
    board.display();
    return;
  }

  // =================================================================
  // fen / setboard: set board from FEN string
  // =================================================================
  if (command == "fen" || command == "setboard") {
    if (parts.length < 2) {
      print("Usage: $command <fen_string>");
      return;
    }
    // Reconstruct FEN string (parts[1] onwards)
    String fenString = parts.sublist(1).join(' ');
    // The C++ readFen is more complex, handling files.
    // We need a `setupFen` equivalent that takes a string.
    // For now, let's assume a simplified `readFenString` exists.
    // The C++ `setupFen` is in `kennySetup.cpp`.
    // Let's create a `readFenString` in `kenny_read_fen.dart`.
    if (readFenString(fenString)) {
      print("Board set from FEN: $fenString");
      board.display();
    } else {
      print("Error: Invalid FEN string.");
    }
    return;
  }

  // =================================================================
  // sd: set search depth
  // =================================================================
  if (command == "sd") {
    if (parts.length < 2) {
      print("Usage: sd <depth>");
      return;
    }
    int? depth = parseInt(parts[1]);
    if (depth == null || depth < 1 || depth > MAX_PLY) {
      print(
        "Invalid search depth: ${parts[1]}. Must be between 1 and $MAX_PLY.",
      );
      return;
    }
    board.searchDepth = depth;
    print("Search depth set to ${board.searchDepth}.");
    return;
  }

  // =================================================================
  // st: set search time
  // =================================================================
  if (command == "st") {
    if (parts.length < 2) {
      print("Usage: st <time_in_seconds>");
      return;
    }
    int? timeInSeconds = parseInt(parts[1]);
    if (timeInSeconds == null || timeInSeconds <= 0) {
      print("Invalid search time: ${parts[1]}. Must be a positive integer.");
      return;
    }
    board.maxTime = timeInSeconds * 1000; // Convert to milliseconds
    print("Search time set to ${board.maxTime ~/ 1000} seconds.");
    return;
  }

  // =================================================================
  // time: set current time for engine (Winboard)
  // =================================================================
  if (command == "time") {
    if (parts.length < 2) {
      print("Usage: time <milliseconds>");
      return;
    }
    int? ms = parseInt(parts[1]);
    if (ms == null || ms < 0) {
      print("Invalid time value: ${parts[1]}.");
      return;
    }
    XB_CTIM = ms;
    // The C++ code also updates board.maxTime in timeControl().
    // For now, just set XB_CTIM.
    print("Engine time set to $ms ms.");
    return;
  }

  // =================================================================
  // otim: set opponent time (Winboard)
  // =================================================================
  if (command == "otim") {
    if (parts.length < 2) {
      print("Usage: otim <milliseconds>");
      return;
    }
    int? ms = parseInt(parts[1]);
    if (ms == null || ms < 0) {
      print("Invalid opponent time value: ${parts[1]}.");
      return;
    }
    XB_OTIM = ms;
    print("Opponent time set to $ms ms.");
    return;
  }

  // =================================================================
  // protover: Winboard protocol version
  // =================================================================
  if (command == "protover") {
    if (parts.length < 2) {
      print("Usage: protover <version>");
      return;
    }
    int? version = parseInt(parts[1]);
    if (version != null && version >= 2) {
      print("feature ping=1 setboard=1 colors=0 usermove=1 san=0 done=1");
      // Other features can be added based on Winboard protocol support
    }
    return;
  }

  // =================================================================
  // xboard: put the engine into "xboard mode"
  // =================================================================
  if (command == "xboard") {
    print(''); // Empty line as per protocol
    XB_COMPUTER_SIDE = XB_NONE;
    XB_MODE = true;
    XB_POST = false;
    board.init();
    print("Entered XBoard mode.");
    return;
  }

  // =================================================================
  // quit: exit the program
  // =================================================================
  if (command == "quit") {
    print("Exiting Kenny. Goodbye!");
    exit(0); // Terminate the Dart application
  }

  // =================================================================
  // Try to interpret as a move (e.g., e2e4, e7e8q)
  // =================================================================
  // This part is complex as it requires move validation and execution.
  // The C++ `isValidTextMove` is used here.
  // For now, a simplified placeholder.
  Move parsedMove = Move();
  // Assuming isValidTextMove is implemented and populates `parsedMove`
  // and `makeMove` updates the board state.
  // This will require the `isValidTextMove` function to be translated first.
  // For now, assume a dummy check.
  bool isValid = false; // Placeholder
  // If `isValidTextMove` is translated:
  // isValid = isValidTextMove(commandLine, parsedMove);

  if (isValid) {
    // makeMove(parsedMove); // This should be handled by the search process.
    // The C++ `commands` function calls `makeMove` if the input is a valid move.
    print(
      "Move $commandLine received. (Move making logic not fully translated yet)",
    );
    // TODO: Implement actual move making.
  } else {
    // =================================================================
    // unknown command:
    // =================================================================
    print("Error: unknown command: $commandLine");
  }
}
