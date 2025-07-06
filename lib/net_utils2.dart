import 'dart:math' as math;
import 'package:dart_torch/nn/module.dart';
import 'package:dart_torch/nn/layer.dart';
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';

// --- Utility Functions ---

// Implements the Clipped ReLU activation function: y = min(max(x, 0), 1)
Value clippedReLU(Value x) {
  // Use the existing Value operations to build the clipped ReLU
  // Assuming Value.clamp(min, max) exists or can be added to value.dart
  return x.relu().clamp(Value(0.0), Value(1.0));
}

// Extension to add clamp to Value (if not already present from previous context)
// This might conflict if Value already has a clamp method.
// For this example, let's assume it's added to Value class or a custom extension for it.
// extension ValueClamp on Value {
//   Value clamp(Value minVal, Value maxVal) {
//     final clamped = Value(data.clamp(minVal.data, maxVal.data).toDouble(), {this, minVal, maxVal}, 'clamp');
//     clamped._backward = () {
//       if (data >= minVal.data && data <= maxVal.data) {
//         grad += clamped.grad;
//       }
//     };
//     return clamped;
//   }

//   // Add sigmoid function to Value for WDL conversion
//   // y = 1 / (1 + exp(-x))
//   Value sigmoid() {
//     final expNegX = (-this).exp(); // Using Value.exp() if implemented
//     final out = Value(1.0 / (1.0 + expNegX.data), {this, expNegX}, 'sigmoid');
//     out._backward = () {
//       // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
//       // out.data is already sigmoid(this.data)
//       grad += out.grad * out.data * (1.0 - out.data);
//     };
//     return out;
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
    // Accumulate columns for active features.
    // This part effectively performs the sparse matrix-vector multiplication.
    for (int featureIdx in activeFeatures) {
      if (featureIdx < 0 || featureIdx >= numInputs) {
        // Handle out-of-bounds features gracefully, though in real usage they should be valid.
        continue;
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
      if (featureIdx < 0 || featureIdx >= numInputs) continue;
      for (int i = 0; i < numOutputsPerPerspective; ++i) {
        newAccumulatorPerspective[i] -= weights[featureIdx][i];
      }
    }
    // Add weights of added features
    for (int featureIdx in addedFeatures) {
      if (featureIdx < 0 || featureIdx >= numInputs) continue;
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

  // To simulate previous state for incremental updates
  BoardState? _previousState;
  // This would typically be passed during move generation or retrieved from search stack
  List<int> _cachedRemovedFeaturesWhite = [];
  List<int> _cachedAddedFeaturesWhite = [];
  List<int> _cachedRemovedFeaturesBlack = [];
  List<int> _cachedAddedFeaturesBlack = [];

  BoardState({
    required this.piecesOnBoard,
    required this.whiteKingSquare,
    required this.blackKingSquare,
    required this.sideToMove,
    BoardState? previousState,
  }) : _previousState = previousState {
    if (previousState != null) {
      // In a real engine, this is where you'd compute the diff
      // For this example, we'll just set dummy diffs
      if (sideToMove == WHITE) {
        _cachedRemovedFeaturesWhite = [1000]; // Dummy feature index
        _cachedAddedFeaturesWhite = [2000]; // Dummy feature index
      } else {
        _cachedRemovedFeaturesBlack = [3000]; // Dummy feature index
        _cachedAddedFeaturesBlack = [4000]; // Dummy feature index
      }
    }
  }

  // Simulates getting active features for a given perspective (for refresh)
  // In a real engine, this would iterate through pieces and their squares.
  List<int> getActiveFeatures(int perspective) {
    final List<int> features = [];
    int currentKingSq = (perspective == WHITE)
        ? whiteKingSquare
        : blackKingSquare;

    // Simulate HalfKP feature generation
    // Each feature is (our_king_square, piece_square, piece_type, piece_color)
    piecesOnBoard.forEach((square, pieceInfo) {
      final pieceType = pieceInfo['type']!;
      final pieceColor = pieceInfo['color']!;

      // Exclude kings for HalfKP (as per the HalfKP description)
      if (pieceType == KING) return;

      // HalfKP Index calculation (simplified for this dummy)
      // The formula: halfkp_idx = piece_square + (p_idx + king_square * 10) * 64
      // p_idx = piece_type * 2 + piece_color (piece_color is "us" or "them", here its absolute color)
      final p_idx =
          pieceType * 2 +
          pieceColor; // Assuming pieceColor is 0 for white, 1 for black
      final halfkp_idx = square + (p_idx + currentKingSq * 10) * 64;

      // Ensure feature index is within expected bounds if any are passed
      if (halfkp_idx >= 0 && halfkp_idx < 40960) {
        // Max HalfKP features is 40960
        features.add(halfkp_idx);
      }
    });

    // Add some random features to make it slightly more dynamic for testing
    // In a real scenario, this is deterministic based on board.
    final random = math.Random();
    for (int i = 0; i < 5; i++) {
      // Add 5 random active features
      features.add(random.nextInt(40960));
    }

    return features.toSet().toList(); // Ensure unique features
  }

  // Determines if a full accumulator refresh is needed (e.g., king moved)
  bool needsRefresh(int perspective) {
    if (_previousState == null)
      return true; // Always refresh for the very first position

    int prevKingSq = (perspective == WHITE)
        ? _previousState!.whiteKingSquare
        : _previousState!.blackKingSquare;
    int currKingSq = (perspective == WHITE) ? whiteKingSquare : blackKingSquare;

    return prevKingSq != currKingSq; // Refresh if king moved
  }

  List<int> getRemovedFeatures(int perspective) {
    return (perspective == WHITE)
        ? _cachedRemovedFeaturesWhite
        : _cachedRemovedFeaturesBlack;
  }

  List<int> getAddedFeatures(int perspective) {
    return (perspective == WHITE)
        ? _cachedAddedFeaturesWhite
        : _cachedAddedFeaturesBlack;
  }

  // Simulate applying a move to get a new board state for demonstration
  BoardState applyMove(String move) {
    // In a real engine, this would parse the move, update piecesOnBoard,
    // and potentially change king positions, sideToMove, etc.
    final newPieces = Map<int, Map<String, int>>.from(piecesOnBoard);
    int newWhiteKingSq = whiteKingSquare;
    int newBlackKingSq = blackKingSquare;
    int newSideToMove = (sideToMove == WHITE) ? BLACK : WHITE;

    // Dummy move logic to demonstrate update/refresh
    if (move == 'e4') {
      // White pawn move
      newPieces.removeWhere((sq, piece) => sq == 10 && piece['type'] == PAWN);
      newPieces[18] = {'type': PAWN, 'color': WHITE}; // New square for pawn
    } else if (move == 'Ke2') {
      // White king move, forces refresh for white
      newWhiteKingSq = 12; // Example king move
    }

    return BoardState(
      piecesOnBoard: newPieces,
      whiteKingSquare: newWhiteKingSq,
      blackKingSquare: newBlackKingSq,
      sideToMove: newSideToMove,
      previousState: this, // Link to previous state for diff calculation
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

  // Scaling factor for CP-space to WDL-space conversion
  final Value scalingFactor;

  NNUEModel({
    required int numFeatures, // N in FeatureSet[N]->M*2->K->1
    required int numOutputsPerPerspective, // M in FeatureSet[N]->M*2->K->1
    required int hiddenLayer1Size, // K in M*2->K
    required int hiddenLayer2Size, // N in K->N
    double wdlScalingFactor = 410.0, // Stockfish value
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
       scalingFactor = Value(wdlScalingFactor);

  // Performs the forward pass given the current board state and an accumulator instance.
  // The accumulator `currentAccumulator` is updated in place.
  Value forward(BoardState currentBoard, NnueAccumulator currentAccumulator) {
    // 1. Update Accumulators for both perspectives
    // In a real search, the previous_accumulator would be from the parent node on the search stack.
    // We assume `currentAccumulator` here is either new or copied from previous and needs updating.

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
            .WHITE], // Assuming current is also previous for update path
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
            .BLACK], // Assuming current is also previous for update path
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
    Value finalEvaluationCP = outputLayer
        .forward(finalInputToOutput)
        .values
        .first;

    // Apply the final scaling as mentioned in the text
    // "The output of the output layer is divided by FV_SCALE = 16"
    return finalEvaluationCP / Value(16.0);
  }

  // Helper to convert CP evaluation to WDL space (for loss calculation)
  Value convertToWDL(Value cpEvaluation) {
    return (cpEvaluation / scalingFactor).sigmoid();
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
  print("--- NNUE Model Implementation in Dart with Loss Concepts ---");

  // NNUE Architecture Parameters (Example from text: 40960 features -> 256*2 -> 32 -> 32 -> 1)
  final numFeatures = 40960; // N in FeatureSet[N]->M*2->K->1
  final numOutputsPerPerspective = 256; // M in FeatureSet[N]->M*2->K->1
  final hiddenLayer1Size = 32; // K in M*2->K
  final hiddenLayer2Size = 32; // N in K->N
  final wdlScalingFactor = 410.0; // From Stockfish examples

  final nnueModel = NNUEModel(
    numFeatures: numFeatures,
    numOutputsPerPerspective: numOutputsPerPerspective,
    hiddenLayer1Size: hiddenLayer1Size,
    hiddenLayer2Size: hiddenLayer2Size,
    wdlScalingFactor: wdlScalingFactor,
  );

  // --- Initial Board State and Accumulator ---
  final initialBoard = BoardState(
    piecesOnBoard: {
      0: {'type': BoardState.ROOK, 'color': BoardState.WHITE},
      1: {'type': BoardState.KNIGHT, 'color': BoardState.WHITE},
      10: {'type': BoardState.PAWN, 'color': BoardState.WHITE},
      63: {'type': BoardState.ROOK, 'color': BoardState.BLACK},
      62: {'type': BoardState.KNIGHT, 'color': BoardState.BLACK},
      53: {'type': BoardState.PAWN, 'color': BoardState.BLACK},
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

  // The accumulator for the current position
  final NnueAccumulator currentPositionAccumulator = NnueAccumulator(
    numOutputsPerPerspective,
  );

  // Perform an initial forward pass
  final nnueEvaluationCP = nnueModel.forward(
    initialBoard,
    currentPositionAccumulator,
  );
  print(
    "Initial NNUE Evaluation (CP-space): ${nnueEvaluationCP.data.toStringAsFixed(4)}",
  );

  // Convert to WDL space for loss calculation
  final nnueEvaluationWDL = nnueModel.convertToWDL(nnueEvaluationCP);
  print(
    "Initial NNUE Evaluation (WDL-space): ${nnueEvaluationWDL.data.toStringAsFixed(4)}",
  );

  // --- Dummy Training Step ---
  print("\n--- Dummy Training Step ---");

  // Example: Target from dataset (e.g., true evaluation in CP) and game result
  final targetEvaluationCP = Value(
    200.0,
  ); // Example target CP (white is up material)
  final gameResultWDL = Value(1.0); // Example: Game resulted in a win for White

  // Convert target CP to WDL space
  final targetEvaluationWDL = nnueModel.convertToWDL(targetEvaluationCP);

  // Interpolation parameter (lambda_)
  final lambda_ = Value(0.7); // 70% evaluation, 30% game result

  // Calculate the combined target for loss
  final combinedTargetWDL =
      lambda_ * targetEvaluationWDL + (Value(1.0) - lambda_) * gameResultWDL;
  print(
    "Combined Target (WDL-space): ${combinedTargetWDL.data.toStringAsFixed(4)}",
  );

  // MSE Loss (exponent 2.0 for simplicity, text suggests >2 like 2.6)
  Value loss = (nnueEvaluationWDL - combinedTargetWDL).pow(2.0);

  print("Initial Loss: ${loss.data.toStringAsFixed(4)}");

  // Zero gradients, perform backward pass
  nnueModel.zeroGrad();
  loss.backward();

  // Optimizer step
  final optimizer = SGD(
    nnueModel.parameters(),
    0.0001,
  ); // Smaller learning rate for stability
  optimizer.step();

  // --- Simulate a Move and Re-evaluate ---
  print("\n--- After 1 training step and a dummy move ---");
  final nextBoardState = initialBoard.applyMove(
    'Ke2',
  ); // Example king move to force refresh
  final NnueAccumulator nextPositionAccumulator = NnueAccumulator(
    numOutputsPerPerspective,
  ); // New accumulator for new position

  final nnueEvaluationCPAfterUpdate = nnueModel.forward(
    nextBoardState,
    nextPositionAccumulator,
  );
  print(
    "NNUE Evaluation (CP-space) After Update: ${nnueEvaluationCPAfterUpdate.data.toStringAsFixed(4)}",
  );

  final nnueEvaluationWDLAfterUpdate = nnueModel.convertToWDL(
    nnueEvaluationCPAfterUpdate,
  );
  print(
    "NNUE Evaluation (WDL-space) After Update: ${nnueEvaluationWDLAfterUpdate.data.toStringAsFixed(4)}",
  );

  Value lossAfterUpdate = (nnueEvaluationWDLAfterUpdate - combinedTargetWDL)
      .pow(2.0);
  print("Loss After Update: ${lossAfterUpdate.data.toStringAsFixed(4)}");
  print(
    "\nThis demonstrates the conceptual flow of NNUE inference with incremental updates",
  );
  print("and a basic training step using WDL conversion and MSE loss.");
  print(
    "Note: Feature factorization itself is a training-time concept (data generation/forward pass modification),",
  );
  print("and the deployed model already has coalesced weights.");
}
