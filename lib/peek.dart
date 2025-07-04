/// kenny_peek.dart
///
/// This file handles checking for user input and time limits during the search.
/// It translates `Board::readClockAndInput()` from kennyPeek.cpp.
/// In a Dart console application, this would involve non-blocking I/O or
/// checking a flag set by an external event loop.
/// For a web-based Dart application, this would be handled by event listeners.

import 'dart:io'; // For stdin, stdout (console I/O)
import 'defs.dart';
import 'board.dart';
import 'commands.dart'; // For processCommand

// Global buffer for console commands (from kennyGlobals.h)
// In Dart, we'll use a more idiomatic way for input.
// For now, let's assume a simple input buffer.
String CMD_BUFF = '';
int CMD_BUFF_COUNT = 0;

/// Checks if we need to stop the search due to time running out or user input.
/// Translates `Board::readClockAndInput()` from kennyPeek.cpp.
void readClockAndInput() {
  // Reset countdown
  board.countdown = UPDATEINTERVAL;

  // Check for time limit
  if (!XB_NO_TIME_LIMIT &&
      (board.timer.getms() - board.msStart) > board.maxTime) {
    board.timedout = true;
    return;
  }

  // Check for user input (non-blocking)
  // In a real Dart console app, you'd use `stdin.listen` or similar.
  // For this simulation, we'll assume `_checkUserInput()` is a non-blocking check.
  _checkUserInput();
}

/// Internal helper to check for pending user input.
/// This simulates `_kbhit()` and `ReadFile()` from the C++ code.
/// In a real Dart application, this would be part of an event loop.
void _checkUserInput() {
  // This is a simplified placeholder.
  // In a true interactive console, you might need to set `stdin.echoMode = false`
  // and `stdin.lineMode = false` to read individual characters immediately.
  // For a basic simulation, we'll check if there's anything in a conceptual buffer.

  // If there's a pending command (e.g., from a previous `stdin.readLineSync`
  // that wasn't fully processed, or if a Winboard command was queued), process it.
  // For a simple console, we can't truly peek without blocking.
  // In a real Winboard implementation, the engine would listen on stdin asynchronously.

  // The C++ `peek` function reads from stdin without blocking.
  // Dart's `stdin.readLineSync()` is blocking.
  // For a non-blocking check, a separate isolate or an event-driven approach
  // with `stdin.listen` would be needed, which is outside the scope of
  // direct function translation for a single-threaded model.

  // For now, we'll simulate processing a command if `CMD_BUFF` has content.
  // In a real scenario, this `CMD_BUFF` would be filled by an external input handler.
  if (CMD_BUFF_COUNT > 0) {
    String command = CMD_BUFF.trim();
    CMD_BUFF = ''; // Clear buffer
    CMD_BUFF_COUNT = 0;

    // Do not stop thinking/pondering/analyzing for certain commands:
    if (command == "." ||
        command == "?" ||
        command == "bk" ||
        command == "hint" ||
        command == "nopost") {
      // These commands do not interrupt the search.
      if (command == "easy") {
        XB_PONDER = false;
      }
      // Process the command but don't set timedout
      // processCommand(command); // This might be too heavy for a peek
      return;
    }

    // Handle time updates (otim, time)
    if (command.startsWith("otim")) {
      int? ms = int.tryParse(command.substring(4).trim());
      if (ms != null) XB_OTIM = ms;
      return;
    }
    if (command.startsWith("time")) {
      int? ms = int.tryParse(command.substring(4).trim());
      if (ms != null) XB_CTIM = ms;
      return;
    }

    // For any other command, assume it's an interrupt
    board.timedout = true;
    // You might want to store the command to be processed after the search stops.
  }
}
