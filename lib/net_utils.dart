import 'dart:math' as math;
import 'package:dart_torch/nn/module.dart';
import 'package:dart_torch/nn/layer.dart';
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';


// --- Utility Functions ---

// Implements the Clipped ReLU activation function: y = min(max(x, 0), 1)
Value clippedReLU(Value x) {
  // Use the existing Value operations to build the clipped ReLU
  return x.relu().clamp(Value(0.0), Value(1.0));
}

// Extension to add clamp to Value (if not already present from previous context)
// This might conflict if Value already has a clamp method.
// Assuming Value.clamp(min, max) exists or can be added to value.dart
// For this example, let's assume it's added to Value class or a custom extension for it.
// extension ValueClamp on Value {
//   Value clamp(Value minVal, Value maxVal) {
//     final clamped = Value(data.clamp(minVal.data, maxVal.data).toDouble(), {
//       this,
//       minVal,
//       maxVal,
//     }, 'clamp');
//     clamped._backward = () {
//       if (data >= minVal.data && data <= maxVal.data) {
//         grad += clamped.grad;
//       }
//     };
//     return clamped;
//   }
// }

// --- NNUE Specific Classes ---

// Represents the weights and biases of the large first layer (L_0)
// This layer's output is the accumulator.
class NnueFirstLayerWeights extends Module {
  final int numInputs; // N in N->M
  final int numOutputsPerPerspective; // M in N->M, e.g., 256 for HalfKP

  // Weights matrix: [input_feature_index][output_neuron_index]
  // In a real NNUE, these would be loaded from a trained model.
  final List<List<Value>> weights;
  final List<Value> biases; // Biases for the M output neurons

  NnueFirstLayerWeights(this.numInputs, this.numOutputsPerPerspective)
    : weights = List.generate(
        numInputs,
        (_) => List.generate(
          numOutputsPerPerspective,
          (_) => Value(math.Random().nextDouble() * 0.01),
        ),
      ), // Small random weights
      biases = List.generate(
        numOutputsPerPerspective,
        (_) => Value(0.0),
      ); // Zero biases

  // Refreshes the accumulator for a given perspective
  // activeFeatures: indices of currently active features for this perspective
  void refreshAccumulator(
    List<int> activeFeatures,
    List<Value> accumulatorPerspective,
  ) {
    // Initialize with biases
    for (int i = 0; i < numOutputsPerPerspective; ++i) {
      accumulatorPerspective[i] = biases[i];
    }
    // Accumulate columns for active features
    for (int featureIdx in activeFeatures) {
      if (featureIdx < 0 || featureIdx >= numInputs) {
        throw ArgumentError("Feature index out of bounds: $featureIdx");
      }
      for (int i = 0; i < numOutputsPerPerspective; ++i) {
        accumulatorPerspective[i] += weights[featureIdx][i];
      }
    }
  }

  // Updates the accumulator based on changes from a previous state
  void updateAccumulator(
    List<Value> newAccumulatorPerspective,
    List<Value> prevAccumulatorPerspective,
    List<int> removedFeatures,
    List<int> addedFeatures,
  ) {
    // Copy previous values as starting point
    for (int i = 0; i < numOutputsPerPerspective; ++i) {
      newAccumulatorPerspective[i] = prevAccumulatorPerspective[i];
    }
    // Subtract weights of removed features
    for (int featureIdx in removedFeatures) {
      if (featureIdx < 0 || featureIdx >= numInputs) {
        throw ArgumentError("Removed feature index out of bounds: $featureIdx");
      }
      for (int i = 0; i < numOutputsPerPerspective; ++i) {
        newAccumulatorPerspective[i] -= weights[featureIdx][i];
      }
    }
    // Add weights of added features
    for (int featureIdx in addedFeatures) {
      if (featureIdx < 0 || featureIdx >= numInputs) {
        throw ArgumentError("Added feature index out of bounds: $featureIdx");
      }
      for (int i = 0; i < numOutputsPerPerspective; ++i) {
        newAccumulatorPerspective[i] += weights[featureIdx][i];
      }
    }
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    for (var row in weights) {
      params.addAll(row);
    }
    params.addAll(biases);
    return params;
  }
}

// Represents the NNUE accumulator state for a given position
class NnueAccumulator {
  // v[0] for white's perspective, v[1] for black's perspective
  final List<List<Value>> v; // [perspective_index][neuron_index]
  final int numOutputsPerPerspective;

  NnueAccumulator(this.numOutputsPerPerspective)
    : v = List.generate(
        2,
        (_) => List.generate(numOutputsPerPerspective, (_) => Value(0.0)),
      );

  // Accessor for a specific perspective
  List<Value> operator [](int perspective) {
    if (perspective != 0 && perspective != 1) {
      throw ArgumentError("Perspective must be 0 (white) or 1 (black).");
    }
    return v[perspective];
  }
}

// Placeholder for Board State and Feature Generation
class BoardState {
  // Constants for piece types and colors (simplified)
  static const int PAWN = 0;
  static const int KNIGHT = 1;
  static const int BISHOP = 2;
  static const int ROOK = 3;
  static const int QUEEN = 4;
  static const int KING = 5;

  static const int WHITE = 0;
  static const int BLACK = 1;

  // Dummy representation of pieces on the board for feature generation
  // Map: square -> {piece_type, piece_color}
  Map<int, Map<String, int>> piecesOnBoard;
  int whiteKingSquare;
  int blackKingSquare;
  int sideToMove; // 0 for WHITE, 1 for BLACK

  BoardState({
    required this.piecesOnBoard,
    required this.whiteKingSquare,
    required this.blackKingSquare,
    required this.sideToMove,
  });

  // Simulates getting active features for a given perspective
  // In a real engine, this would iterate through pieces and their squares.
  List<int> getActiveFeatures(int perspective) {
    final List<int> features = [];
    int currentKingSq = (perspective == WHITE)
        ? whiteKingSquare
        : blackKingSquare;

    // Simulate HalfKP feature generation
    // Each feature is (our_king_square, piece_square, piece_type, piece_color)
    // Here, piece_color refers to its actual color, not "us" or "them".
    piecesOnBoard.forEach((square, pieceInfo) {
      final pieceType = pieceInfo['type']!;
      final pieceColor = pieceInfo['color']!;

      // Exclude kings for HalfKP
      if (pieceType == KING) return;

      // HalfKP Index calculation (simplified for this dummy)
      // p_idx = piece_type * 2 + piece_color
      // halfkp_idx = piece_square + (p_idx + king_square * 10) * 64
      // This formula is for a single piece. We need to map (piece_type, piece_color) to a compact int for p_idx.
      final p_idx = pieceType * 2 + pieceColor; // Simplified mapping
      final halfkp_idx = square + (p_idx + currentKingSq * 10) * 64;

      features.add(halfkp_idx);
    });
    return features;
  }

  // Dummy methods for move handling - these would be complex in a real engine
  bool needsRefresh(int perspective) {
    // Only refresh if king moved from previous position for that perspective
    // Simplified: always refresh for this dummy example or never.
    return true; // For demonstration, let's say it always needs refresh.
  }

  // In a real engine, these would compare current and previous board states
  List<int> getRemovedFeatures(int perspective) => []; // Dummy
  List<int> getAddedFeatures(int perspective) => []; // Dummy

  // Simulate applying a move to get a new board state for demonstration
  BoardState applyMove(String move) {
    // In a real engine, this would parse the move, update piecesOnBoard,
    // and potentially change king positions, sideToMove, etc.
    // For now, just return a new state with slightly altered king squares to show refresh/update paths.
    return BoardState(
      piecesOnBoard: Map.from(piecesOnBoard), // Copy pieces
      whiteKingSquare: whiteKingSquare + (move == 'e4' ? 1 : 0), // Dummy change
      blackKingSquare: blackKingSquare + (move == 'e5' ? 1 : 0), // Dummy change
      sideToMove: (sideToMove == WHITE) ? BLACK : WHITE,
    );
  }
}

// --- The NNUE Model ---

class NNUEModel extends Module {
  // L_0 weights (feature transformer)
  final NnueFirstLayerWeights ft;

  // Hidden Layer 1 (L_1)
  final Layer hiddenLayer1;
  // Hidden Layer 2 (L_2)
  final Layer hiddenLayer2;
  // Output Layer (L_3)
  final Layer outputLayer;

  final int M; // numOutputsPerPerspective from ft (e.g., 256)
  final int K; // Output size of L_1 (e.g., 32)
  final int N; // Output size of L_2 (e.g., 32)

  NNUEModel({
    required int numFeatures, // N in N->M
    required int numOutputsPerPerspective, // M in N->M
    required int hiddenLayer1Size, // K in M*2->K
    required int
    hiddenLayer2Size, // N in K->N (note: text uses N for L2 input, K for L1 input)
  }) : ft = NnueFirstLayerWeights(numFeatures, numOutputsPerPerspective),
       hiddenLayer1 = Layer.fromNeurons(
         2 * numOutputsPerPerspective,
         hiddenLayer1Size,
       ), // M*2 inputs
       hiddenLayer2 = Layer.fromNeurons(
         hiddenLayer1Size,
         hiddenLayer2Size,
       ), // K inputs
       outputLayer = Layer.fromNeurons(
         hiddenLayer2Size,
         1,
       ), // N inputs, 1 output
       M = numOutputsPerPerspective,
       K = hiddenLayer1Size,
       N = hiddenLayer2Size;

  // Performs the forward pass given the current board state and an accumulator instance.
  // The accumulator `currentAccumulator` is updated in place.
  Value forward(BoardState currentBoard, NnueAccumulator currentAccumulator) {
    // 1. Update Accumulators for both perspectives
    // In a real search, the previous_accumulator would be from the parent node on the search stack.
    // For this example, we'll simplify and always refresh or use dummy updates.

    // White's perspective
    if (currentBoard.needsRefresh(BoardState.WHITE)) {
      ft.refreshAccumulator(
        currentBoard.getActiveFeatures(BoardState.WHITE),
        currentAccumulator[BoardState.WHITE],
      );
    } else {
      ft.updateAccumulator(
        currentAccumulator[BoardState.WHITE],
        currentAccumulator[BoardState
            .WHITE], // Using same for prev_acc in dummy
        currentBoard.getRemovedFeatures(BoardState.WHITE),
        currentBoard.getAddedFeatures(BoardState.WHITE),
      );
    }

    // Black's perspective
    if (currentBoard.needsRefresh(BoardState.BLACK)) {
      ft.refreshAccumulator(
        currentBoard.getActiveFeatures(BoardState.BLACK),
        currentAccumulator[BoardState.BLACK],
      );
    } else {
      ft.updateAccumulator(
        currentAccumulator[BoardState.BLACK],
        currentAccumulator[BoardState
            .BLACK], // Using same for prev_acc in dummy
        currentBoard.getRemovedFeatures(BoardState.BLACK),
        currentBoard.getAddedFeatures(BoardState.BLACK),
      );
    }

    // 2. Combine Accumulators based on side to move (tempo)
    final List<Value> combinedAccData = [];
    if (currentBoard.sideToMove == BoardState.WHITE) {
      combinedAccData.addAll(currentAccumulator[BoardState.WHITE]);
      combinedAccData.addAll(currentAccumulator[BoardState.BLACK]);
    } else {
      combinedAccData.addAll(currentAccumulator[BoardState.BLACK]);
      combinedAccData.addAll(currentAccumulator[BoardState.WHITE]);
    }
    final ValueVector combinedAccumulator = ValueVector(combinedAccData);

    // 3. Clipped ReLU on combined accumulator (input to L_1)
    ValueVector l1Input = ValueVector(
      combinedAccumulator.values.map((v) => clippedReLU(v)).toList(),
    );

    // 4. Linear Layer 1 (L_1)
    ValueVector l1Output = hiddenLayer1.forward(l1Input);

    // 5. Clipped ReLU on L_1 output (input to L_2)
    ValueVector l2Input = ValueVector(
      l1Output.values.map((v) => clippedReLU(v)).toList(),
    );

    // 6. Linear Layer 2 (L_2)
    ValueVector l2Output = hiddenLayer2.forward(l2Input);

    // 7. Clipped ReLU on L_2 output (input to final output layer)
    ValueVector finalInputToOutput = ValueVector(
      l2Output.values.map((v) => clippedReLU(v)).toList(),
    );

    // 8. Output Layer (L_3)
    Value finalEvaluation = outputLayer
        .forward(finalInputToOutput)
        .values
        .first;

    // The text states the final output is divided by FV_SCALE = 16
    return finalEvaluation / Value(16.0);
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    params.addAll(ft.parameters());
    params.addAll(hiddenLayer1.parameters());
    params.addAll(hiddenLayer2.parameters());
    params.addAll(outputLayer.parameters());
    return params;
  }
}

// SGD class (from your encoder_example.dart)
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("--- NNUE Model Implementation in Dart ---");

  // NNUE Architecture Parameters (Example from text: 40960 features -> 256*2 -> 32 -> 32 -> 1)
  // M = 256, K = 32, N = 32
  final numFeatures = 40960; // N in FeatureSet[N]->M*2->K->1
  final numOutputsPerPerspective = 256; // M in FeatureSet[N]->M*2->K->1
  final hiddenLayer1Size = 32; // K in M*2->K
  final hiddenLayer2Size = 32; // N in K->N

  final nnueModel = NNUEModel(
    numFeatures: numFeatures,
    numOutputsPerPerspective: numOutputsPerPerspective,
    hiddenLayer1Size: hiddenLayer1Size,
    hiddenLayer2Size: hiddenLayer2Size,
  );

  // Initialize a dummy board state and accumulator
  final initialBoard = BoardState(
    piecesOnBoard: {
      0: {'type': BoardState.ROOK, 'color': BoardState.WHITE},
      1: {'type': BoardState.KNIGHT, 'color': BoardState.WHITE},
      10: {'type': BoardState.PAWN, 'color': BoardState.WHITE},
      63: {'type': BoardState.ROOK, 'color': BoardState.BLACK},
      62: {'type': BoardState.KNIGHT, 'color': BoardState.BLACK},
      53: {'type': BoardState.PAWN, 'color': BoardState.BLACK},
      // Example: A king on a specific square
      20: {
        'type': BoardState.KING,
        'color': BoardState.WHITE,
      }, // White King on square 20
      45: {
        'type': BoardState.KING,
        'color': BoardState.BLACK,
      }, // Black King on square 45
    },
    whiteKingSquare: 20,
    blackKingSquare: 45,
    sideToMove: BoardState.WHITE,
  );

  // The accumulator will be managed on the "search stack"
  final NnueAccumulator currentPositionAccumulator = NnueAccumulator(
    numOutputsPerPerspective,
  );

  // Perform a forward pass (initial evaluation)
  final nnueEvaluation = nnueModel.forward(
    initialBoard,
    currentPositionAccumulator,
  );
  print("Initial NNUE Evaluation: ${nnueEvaluation.data.toStringAsFixed(4)}");

  // --- Dummy Training Step ---
  print("\n--- Dummy Training Step for NNUE ---");

  // Dummy target evaluation (e.g., from a dataset of known positions)
  final targetEvaluation = Value(0.75); // Example target

  // Simple squared error loss
  Value dummyLoss = (targetEvaluation - nnueEvaluation).pow(2);

  print("Initial Dummy Loss: ${dummyLoss.data.toStringAsFixed(4)}");

  // Zero gradients, perform backward pass
  nnueModel.zeroGrad();
  dummyLoss.backward();

  // Optimizer step
  final optimizer = SGD(nnueModel.parameters(), 0.001); // Small learning rate
  optimizer.step();

  // Simulate a move and re-evaluate
  final nextBoardState = initialBoard.applyMove('d4');
  // For simplicity, we'll reuse the same accumulator object,
  // but in a real search tree, each node would have its own accumulator.
  final nnueEvaluationAfterUpdate = nnueModel.forward(
    nextBoardState,
    currentPositionAccumulator,
  );
  Value dummyLossAfterUpdate = (targetEvaluation - nnueEvaluationAfterUpdate)
      .pow(2);

  print(
    "NNUE Evaluation After 1 Move & Update: ${nnueEvaluationAfterUpdate.data.toStringAsFixed(4)}",
  );
  print(
    "Dummy Loss After 1 Move & Update: ${dummyLossAfterUpdate.data.toStringAsFixed(4)}",
  );
  print(
    "\nThis demonstrates the conceptual flow of NNUE with incremental updates and training.",
  );
  print(
    "A complete implementation would require robust board representation, feature generation, and accumulator management on a search stack.",
  );
}
