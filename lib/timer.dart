/// kenny_timer.dart
///
/// This file defines the `Timer` class for measuring time,
/// translating the C++ `Timer` struct and its methods from kennyTimer.h and kennyTimer.cpp.
/// It uses Dart's `DateTime` and `Duration` for time tracking.

import 'defs.dart'; // For U64 typedef

class Timer {
  bool running;
  U64 startTime; // Milliseconds since epoch when timer started/reset
  U64 stopTime; // Milliseconds since epoch when timer stopped
  U64 stopTimeDelta; // Accumulates time when timer was stopped
  // In Dart, we'll use DateTime objects and calculate milliseconds directly.
  // The `timeb` struct from C++ is not directly translatable.

  Timer()
      : running = false,
        startTime = 0,
        stopTime = 0,
        stopTimeDelta = 0;

  /// Initializes (starts) the timer.
  void init() {
    if (!running) {
      running = true;
      // Get current system time in milliseconds since epoch
      startTime = DateTime.now().millisecondsSinceEpoch + stopTimeDelta;
    }
  }

  /// Stops the timer.
  void stop() {
    if (running) {
      running = false;
      stopTime = DateTime.now().millisecondsSinceEpoch;
      stopTimeDelta = startTime - stopTime; // This logic might need adjustment based on C++ intent
      // In C++, stopTimeDelta seems to be used to adjust startTime on subsequent `init` calls
      // to resume timing from where it left off.
    }
  }

  /// Resets the timer.
  void reset() {
    if (running) {
      startTime = DateTime.now().millisecondsSinceEpoch;
    } else {
      startTime = stopTime; // If not running, reset start to where it stopped
      stopTimeDelta = 0; // Clear accumulated stopped time
    }
  }

  /// Returns the elapsed time in milliseconds.
  U64 getms() {
    if (running) {
      return DateTime.now().millisecondsSinceEpoch - startTime;
    } else {
      return stopTime - startTime;
    }
  }

  /// Returns the current system time in milliseconds since epoch.
  U64 getsysms() {
    return DateTime.now().millisecondsSinceEpoch;
  }

  /// Displays time in seconds with 2 decimals.
  /// In Dart, we'll return a formatted string instead of printing directly.
  String display() {
    double elapsedSeconds;
    if (running) {
      elapsedSeconds = (DateTime.now().millisecondsSinceEpoch - startTime) / 1000.0;
    } else {
      elapsedSeconds = (stopTime - startTime) / 1000.0;
    }
    return elapsedSeconds.toStringAsFixed(2);
  }

  /// Displays time in hh:mm:ss format.
  /// In Dart, we'll return a formatted string instead of printing directly.
  String displayhms() {
    U64 totalMilliseconds;
    if (running) {
      totalMilliseconds = DateTime.now().millisecondsSinceEpoch - startTime;
    } else {
      totalMilliseconds = stopTime - startTime;
    }

    int hh = (totalMilliseconds ~/ 1000) ~/ 3600;
    int mm = ((totalMilliseconds ~/ 1000) - hh * 3600) ~/ 60;
    int ss = (totalMilliseconds ~/ 1000) - hh * 3600 - mm * 60;

    return '${hh.toString().padLeft(2, '0')}:${mm.toString().padLeft(2, '0')}:${ss.toString().padLeft(2, '0')}';
  }
}
