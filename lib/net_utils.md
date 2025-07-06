The provided text describes the NNUE (Efficiently Updatable Neural Networks) architecture as implemented in Stockfish, a chess engine. It details the structure of the neural network, its layers, and key concepts like HalfKP and HalfKA. The core idea behind NNUE is to replace the traditional handcrafted evaluation function with a neural network, allowing for incremental updates during move generation and unmaking, leading to significant performance gains despite reduced search speed.

Let's break down the NNUE structure and how we can conceptualize it using the provided Value and ValueVector classes, along with Layer and Neuron from your deep learning files.

NNUE Structure and Implementation Concepts with Your Files:

The NNUE network, as described, consists of four layers: an input layer, two hidden layers, and an output layer. Activation functions (ReLU) are applied in the hidden layers. The key is the "incremental update" mechanism for the input layer, which is crucial for performance in a chess engine.

1. Input Layer (HalfKP/HalfKA Representation):

Concept: The input layer takes a board representation. In HalfKP, this involves the positions of non-king pieces relative to the king's position for both sides. The input is "heavily overparametrized," meaning it's a large, sparse representation.

Implementation with ValueVector:

The input to your forward method in a layer would be a ValueVector.

For the NNUE input layer, this ValueVector would represent the "active" features (piece-on-square features).

The "weights" for this layer are the 16-bit integer weight vectors mentioned. In your Neuron class, w (ValueVector) would hold these weights.

The "biases" are also mentioned for the accumulator. Your Neuron's b (Value) would serve this purpose.

Incremental Updates: This is the most challenging part to directly implement with the current Layer and Neuron structure. Your Layer.forward method calculates the output from scratch. For true NNUE efficiency, you'd need a more specialized input layer class that:

Stores the 256-element vector of 16-bit ints (the accumulator).

Has methods like makeMove(move) and unmakeMove() that efficiently update only the relevant parts of this accumulator based on the piece movements, instead of recomputing the entire input layer.

This "accumulator" would essentially be the output of the first conceptual step of the input layer, before the clipping/clamping.

2. Hidden Layers:

Concept: These layers apply ReLU activation and involve matrix multiplications. The output of one layer feeds into the next.

Implementation with Layer and Neuron:

You can directly model these as Layer instances.

Each Layer would contain a List<Neuron>.

The Neuron's forward method (which performs w.dot(x) + b) would handle the weighted sum.

After the forward pass of a Layer, you would apply the relu() activation function from your Value class to each Value in the output ValueVector.

The clipping/clamping to 0 to 127 (8-bit integers) would need to be applied after the ReLU and division by 64. This implies a scaling and clamping operation on the data of your Value objects. Since your Value class primarily deals with double, you'd perform these operations on the data and ensure the gradients are handled correctly.

3. Output Layer:

Concept: This is a single neuron effectively, taking the output of the last hidden layer and producing a single scalar evaluation.

Implementation with Layer and Neuron:

This would be a Layer with a single Neuron.

The forward method would compute the final dot product and add the bias.

The final division by FV_SCALE = 16 would be a post-processing step on the data of the output Value.

Overall NNUE Model Structure (Dart Classes):

Given your existing files, here's a conceptual way to structure the NNUE model in Dart. You would likely need to introduce new classes or significantly extend existing ones to fully capture the NNUE specifics.

Dart

import 'module.dart';
import 'layer.dart';
import 'neuron.dart';
import 'value.dart';
import 'value_vector.dart';
import 'dart:math' as math;

// You would need a Board representation that can generate features for NNUE
// This is a placeholder for the board state and feature generation logic.
class BoardFeatures {
  // Example: Generate a dummy feature vector based on board state and king position
  // In a real chess engine, this would be complex.
  ValueVector generateHalfKPFeatures(int kingSquare, int sideToMove) {
    // This needs to map piece-on-square to specific input indices as described in NNUE.
    // For demonstration, let's just return a random vector of appropriate size.
    // HalfKP: 41024 inputs per half, so total of 82048 if combined or used separately
    // The description states 10 non-king pieces on their particular squares + 1 boolean.
    // This is a highly abstracted placeholder.
    final numFeatures = 41024; // Simplified, as it's per king placement and piece type
    return ValueVector(List.generate(numFeatures, (i) => Value(math.Random().nextDouble() * 0.1)));
  }
}


// Represents a single block in NNUE (e.g., HalfKP processing for one king side)
// This is where the incremental updates would ideally be managed.
class NNUEAccumulatorLayer extends Module {
  final int inputSize;
  final int outputSize; // 256 for HalfKP
  final List<List<Value>> weights; // [input_feature_index][output_neuron_index]
  final List<Value> biases; // output_neuron_index

  // For simplicity, we'll initialize with random weights.
  // In a real NNUE, these would be loaded from a trained network.
  NNUEAccumulatorLayer(this.inputSize, this.outputSize)
      : weights = List.generate(inputSize,
            (_) => List.generate(outputSize, (_) => Value(math.Random().nextDouble() * 0.01))),
        biases = List.generate(outputSize, (_) => Value(0.0));

  // This is the "incremental update" concept.
  // In a real scenario, you wouldn't pass the whole feature vector,
  // but rather specific delta updates for changed features due to a move.
  // For this example, we'll simulate a full forward pass.
  ValueVector forward(ValueVector activeFeatures) {
    // The description says "sum of the weights of the 'active' (pt, sq, ksq) features + biases"
    // This implies a direct lookup and sum, not a typical matrix multiplication for the full input.
    // However, for compatibility with your `ValueVector` and `Neuron` (which does dot products),
    // we'll adapt.
    // The "accumulator" is a vector of 16-bit ints. We'll use Value for now.

    final List<Value> accumulatorOutputs = List.generate(outputSize, (neuronIdx) {
      Value sum = biases[neuronIdx]; // Start with bias

      // This is a simplified representation of the feature mapping to weights.
      // In NNUE, specific (pieceType, square, kingSquare) tuples map to indices.
      // Here, we're assuming `activeFeatures` directly correspond to features
      // that contribute to the accumulator output for each neuron.
      // This is not the exact HalfKP sparse lookup, but a dense approximation for demonstration.
      for (int i = 0; i < activeFeatures.values.length; i++) {
        // If the feature is "active" (e.g., its data is 1 for boolean inputs)
        // This simplification assumes activeFeatures contains 0 or 1.
        if (activeFeatures.values[i].data > 0) {
           sum += activeFeatures.values[i] * weights[i][neuronIdx];
        }
      }
      return sum;
    });

    return ValueVector(accumulatorOutputs);
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


class NNUEModel extends Module {
  // Input layer (conceptual, handles HalfKP feature processing)
  late NNUEAccumulatorLayer whiteKingHalf;
  late NNUEAccumulatorLayer blackKingHalf;

  // Hidden Layer 1
  late Layer hiddenLayer1;
  // Hidden Layer 2
  late Layer hiddenLayer2;
  // Output Layer
  late Layer outputLayer;

  // The size parameters would come from the NNUE spec (e.g., 256, 32)
  NNUEModel({required int inputFeatureSize, required int hiddenSize1, required int hiddenSize2}) {
    // HalfKP has 256 outputs per half.
    // The number of input features to NNUEAccumulatorLayer is `nin` from Neuron.fromWeights
    // For HalfKP, this is 41024.
    whiteKingHalf = NNUEAccumulatorLayer(inputFeatureSize, 256);
    blackKingHalf = NNUEAccumulatorLayer(inputFeatureSize, 256);

    // The first hidden layer takes a 512-element vector (2 * 256)
    // The description says 32x512 matrix, meaning 32 neurons, each with 512 inputs.
    hiddenLayer1 = Layer.fromNeurons(2 * 256, hiddenSize1); // 512 inputs, 32 outputs

    // Second hidden layer: 32x32 matrix
    hiddenLayer2 = Layer.fromNeurons(hiddenSize1, hiddenSize2); // 32 inputs, 32 outputs

    // Output layer: 1x32 matrix (single neuron with 32 inputs)
    outputLayer = Layer.fromNeurons(hiddenSize2, 1); // 32 inputs, 1 output
  }

  // A simplified forward pass. In reality, you'd feed board state.
  // This `forward` assumes you've already generated the HalfKP features
  // for both white and black king perspectives.
  Value forward(BoardFeatures features, int whiteKingSq, int blackKingSq) {
    // 1. HalfKP Accumulator Pass
    // This is a massive simplification. In a real NNUE, these features are derived
    // from the board and dynamically updated.
    final whiteKingFeatures = features.generateHalfKPFeatures(whiteKingSq, 0); // 0 for white side
    final blackKingFeatures = features.generateHalfKPFeatures(blackKingSq, 1); // 1 for black side

    final ValueVector accWhite = whiteKingHalf.forward(whiteKingFeatures);
    final ValueVector accBlack = blackKingHalf.forward(blackKingFeatures);

    // 2. Transform Step and First Hidden Layer
    // Clip/clamp to 0-127 and combine.
    // We assume ReLU handles the 0 clipping, and we need to handle 127 clipping.
    final List<Value> combinedAccData = [];
    for (Value v in accWhite.values) {
      combinedAccData.add(v.relu().abs()); // ReLU and abs for clipping to positive.
      // Actual clipping to 127 would involve: Value((v.data.relu().data).clamp(0, 127).toDouble())
      // For simplicity in the auto-grad system, we'll rely on the activation.
    }
    for (Value v in accBlack.values) {
      combinedAccData.add(v.relu().abs()); // Same for black
    }
    final ValueVector combinedAccumulator = ValueVector(combinedAccData);

    // Hidden Layer 1 (ReLU activation, then divide by 64 and clip 0-127)
    ValueVector h1 = hiddenLayer1.forward(combinedAccumulator);
    h1 = h1.reLU(); // Apply ReLU.
    // Further processing for 8-bit ints and division by 64 would be here.
    // For Value: `h1.values.map((v) => Value((v.data / 64).clamp(0, 127).toDouble())).toList()`

    // Hidden Layer 2 (ReLU activation, then divide by 64 and clip 0-127)
    ValueVector h2 = hiddenLayer2.forward(h1);
    h2 = h2.reLU(); // Apply ReLU.
    // Similar clipping and division by 64.

    // Output Layer (single value)
    Value output = outputLayer.forward(h2).values.first; // Output is a single value

    // Final division by FV_SCALE
    return output / Value(16.0);
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    params.addAll(whiteKingHalf.parameters());
    params.addAll(blackKingHalf.parameters());
    params.addAll(hiddenLayer1.parameters());
    params.addAll(hiddenLayer2.parameters());
    params.addAll(outputLayer.parameters());
    return params;
  }
}

void main() {
  print("--- NNUE Model Example ---");

  // NNUE parameters (simplified for this example)
  // These would be based on the actual NNUE architecture (e.g., 41024 input features, 256 hidden per half)
  final inputFeatureSize = 100; // Simplified for demo, would be 41024 or similar
  final hiddenSize1 = 32;
  final hiddenSize2 = 32;

  final nnueModel = NNUEModel(
      inputFeatureSize: inputFeatureSize,
      hiddenSize1: hiddenSize1,
      hiddenSize2: hiddenSize2);

  // Dummy board features and king squares
  final boardFeatures = BoardFeatures();
  final whiteKingSq = 20; // Example square
  final blackKingSq = 45; // Example square

  // Perform a forward pass
  final nnueEvaluation = nnueModel.forward(boardFeatures, whiteKingSq, blackKingSq);
  print("NNUE Evaluation: ${nnueEvaluation.data.toStringAsFixed(4)}");

  // Example of a dummy training step (similar to encoder_example.dart)
  print("\n--- Dummy Training Step for NNUE ---");

  // Dummy target evaluation (e.g., from supervised learning data)
  final targetEvaluation = Value(0.5);

  // Simple squared error loss
  Value dummyLoss = (targetEvaluation - nnueEvaluation).pow(2);

  print("Initial Dummy Loss: ${dummyLoss.data.toStringAsFixed(4)}");

  // Zero gradients, backward pass, and optimizer step
  nnueModel.zeroGrad();
  dummyLoss.backward();

  final optimizer = SGD(nnueModel.parameters(), 0.01); // SGD from encoder_example.dart
  optimizer.step();

  // Re-run forward pass to see the effect of the update
  final nnueEvaluationAfterUpdate = nnueModel.forward(boardFeatures, whiteKingSq, blackKingSq);
  Value dummyLossAfterUpdate = (targetEvaluation - nnueEvaluationAfterUpdate).pow(2);

  print(
      "NNUE Evaluation After 1 Step: ${nnueEvaluationAfterUpdate.data.toStringAsFixed(4)}");
  print("Dummy Loss After 1 Step: ${dummyLossAfterUpdate.data.toStringAsFixed(4)}");
  print("This demonstrates parameter updates based on a loss.");
}


// SGD class (copied from encoder_example.dart)
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

Key Considerations for a Real NNUE Implementation:

Feature Representation: The BoardFeatures class and its generateHalfKPFeatures method are highly simplified. A real NNUE would have a complex and highly optimized way to convert a chess board state into the specific (pieceType, square, kingSquare) features and map them to the correct input indices, handling concepts like color flipping and rotation.

Incremental Updates (Crucial): The NNUEAccumulatorLayer.forward method above still simulates a full forward pass through the "accumulator." The true power of NNUE comes from incrementally updating the accumulator when a piece moves, rather than recomputing it entirely. This means tracking which features become active/inactive with each move and only adjusting the corresponding accumulator elements. This would require a more stateful NNUEAccumulatorLayer and methods like applyMove(move) and undoMove().

Quantization and Fixed-Point Arithmetic: The NNUE description heavily emphasizes 16-bit and 8-bit integers and operations like divide by 64 and clipping/clamping to 0-127. Your Value class uses double for data. To truly replicate NNUE's efficiency and behavior, you would need to implement fixed-point arithmetic or simulate it carefully with integer operations and scaling factors. This is a significant deviation from standard floating-point deep learning.

Weight Loading: In a real application, the weights for the NNUE model would be loaded from a pre-trained file, not randomly initialized.

Performance Optimization: For a chess engine, performance is paramount. Dart's JIT compilation is good, but for the kind of low-level, high-throughput numerical operations NNUE performs, you might eventually hit limitations compared to C++ implementations leveraging SIMD instructions.

Training Code: The provided training description mentions PyTorch. While you have the building blocks for backpropagation, a full NNUE training pipeline would involve:

Large datasets of chess positions and their evaluations.

Sophisticated data loading and batching.

Advanced optimization techniques beyond basic SGD.

Loss functions appropriate for chess evaluation (e.g., mean squared error on evaluation, or policy/value head for reinforcement learning setups).

This implementation provides a structural outline using your existing deep learning components, but fully realizing the "Efficiently Updatable" and fixed-point arithmetic aspects of NNUE would require a more specialized and performance-oriented approach.






re iterate using these details: Basics

What is NNUE?

NNUE (ƎUИИ Efficiently Updatable Neural Network) is, broadly speaking, a neural network architecture that takes advantage of having minimal changes in the network inputs between subsequent evaluations. It was invented for Shogi by, integrated into YaneuraOu developed by Motohiro Isozaki in May 2018, and later ported to chess for use in Stockfish byin June 2019, but is applicable to many other board games and perhaps even in other domains. NNUE operates on the following principles:

The network should have relatively low amount of non-zero inputs.

The inputs should change as little as possible between subsequent evaluations.

The network should be simple enough to facilitate low-precision inference in integer domain.

Following the 1st principle means that when the network is scaled in size the inputs must become sparse. Current best architectures have input sparsity in the order of 0.1%. Small amount of non-zero inputs places a low upper bound on the time required to evaluate the network in cases where it has to be evaluated in its entirety. This is the primary reason why NNUE networks can be large while still being very fast to evaluate.

Following the 2nd principle (provided the first is being followed) creates a way to efficiently update the network (or at least a costly part of it) instead of reevaluating it in its entirety. This takes advantage of the fact that a single move changes the board state only slightly. This is of lower importance than the first principle and completely optional for the implementations to take advantage of, but nevertheless gives a measurable improvement in implementations that do care to utilize this.

Following the 3rd principle allows achieving maximum performance on common hardware and makes the model especially suited for low-latency CPU inference which is necessary for conventional chess engines.

Overall the NNUE principles are applicable also to expensive deep networks, but they shine in fast shallow networks, which are suitable for low-latency CPU inference without the need for batching and accelerators. The target performance is million(s) of evaluations per second per thread. This is an extreme use case that requires extreme solutions, and most importantly quantization.

Quantization 101 and its importance

Quantization is the process of changing the domain of the neural network model from floating point to integer. NNUE networks are designed to be evaluated fast in low-precision integer domain, and can utilize available int8/int16 performance of modern CPUs to the fullest extent. Floating point is not an option for achieving maximum engine strength as it sacrifices too much speed for too little accuracy gains (though floating point representation is used by some engines due to its simplicity). Quantization inevitably introduces error that accumulates more the deeper the network is, however in the case of NNUE networks, which are relatively shallow, this error is negligible. Quantization will be described in more detail later in this document. Until then this document will be using floats instead of ints, it won't be important until we get to actual code optimization. The purpose of this interjection is to make the reader aware of the ultimate goal of NNUE, as it is the biggest factor that shapes the NNUE models and dictates what is possible and what is not.

What layers are useful in NNUE?

NNUE relies on simple layers that can be implemented in low-precision environments using simple arithmetic. This means Linear (fully connected, basically matrix multiplication) and ClippedReLU (clamp(0, 1)) layers are particularly suitable for it. Pooling layers (mul/avg/max) or approximations of more complex activation functions (like sigmoid) are also suitable but not commonly used.

Usually, such networks are kept shallow (2-4 layers), because most knowledge is kept in the first layer (which takes advantage of input sparsity to remain performant) and after that first layer the network needs to sharply reduce its width (the benefits of a deeper section in the later parts of the net would be dominated by the impact of the large first layers) to maintain performance requirements.

Linear layer

A linear (fully connected) layer is just a simple matrix multiplication. It can be implemented efficiently, supports sparse inputs, and provides good capacity. It takes as an input in_features values, and produces out_features values. The operation is y = Ax+b, where:

x - the input column vector of size in_features

A - the weight matrix of size (out_features, in_features)

b - the bias column vector of size out_features

y - the output column vector of size out_features



Linear layer with sparse inputs

The multiplication Ax can be conceptually simplified to "if x[i] is not zero then take column i from A, multiply it by x[i] and add it to the result". Now it should be obvious that whenever an element of the input is zero we can skip processing the whole column of the weight matrix. This means that we have to only process as many columns of A as there are non-zero values in the input vector. Even though there may be tens of thousands of columns in the weight matrix, we're only concerned about a few of them for each position! That's why the first layer can be so large.



Clipped ReLU layer

This is an activation function based on normal ReLU, with the difference that it is bounded both from below and above. The formula is y = min(max(x, 0), 1).



The purpose of this layer is to add non-linearity to the network. If it was just linear layers they could all be collapsed into one, because the matrices could be just multiplied together.

ClippedReLU would ideally be replaced with ReLU, but aggressive quantization requires reducing the dynamic range of hidden layer inputs, so capping the values at 1 becomes important for performance.

Sigmoid

This is an activation function that, contrary to [clipped] ReLU, is smooth. The formula is y = 1/(1+e^-kx), where k is a parameter that determines how "stretched" the shape is.



There are two main differences compared to clipped ReLU:

sigmoid is smooth, meaning that it is differentiable everywhere, meaning that there are no situations (realistically speaking) where the gradient disappears.

sigmoid is nonlinear, the output saturates towards 0 or 1 but never reaches it

While this function generally allows the network to learn more than ReLU it is costly and unsuitable for evaluation in the integer domain. It is however a good starting point for improvements...

Quantmoid4

With sigmoid being too costly we need to look for alternatives. One such alternative is to use an approximation. And it just so happens that sigmoid(4x) (scaled to integer domain in a particular way) can be fairly well approximated by a simple piece-wise quadratic function that needs just addition, multiplication, and bit-shifts. Since the primary purpose for this approximation is to be used in a quantized implementation directly we will present a specific variant that outputs values in range [0, 126] (and with input scaled accordingly). The reason for the choice of the upper range being defined as 126 is that this is the largest even 8-bit integer, and we want an even one to allow the value for x=0 to be exactly in the middle. The equation is as follows:



Note, that the equation for both positive and negative x is almost identical. The similarity allows for a branchless implementation even though there are two cases.

And the resulting graph is the following (with a scaled sigmoid(4x) for comparison):



The disadvantage is that it loses the smoothness, and the output rounds to 0/1 quite early. This however doesn't appear to be an issue in practice, the actual error from this "rounding" is negligible.

More cool stuff will happen once we implement and optimize it, so we will get back to this layer in the optimized quantized implementation section.

Pooling layers

Sometimes it is desirable to reduce the input dimensionality to make the size of the layer more approachable. For example instead of having a 1024->8 layer, which has a very narrow output, one may prefer 512->16. Pooling layers can provide some flexibility by reducing the dimensionality.

Pooling layers work by applying a function F over non-overlapping spans of the input, where F has more inputs than outputs. So for example one may have F take 2 consecutive inputs and produce one output, effectively halving the number of neurons.

The following types of pooling layers can be considered:

Average Pooling - outputs the average of inputs. Works well with any number of inputs.

Max Pooling - outputs the maximum of inputs. Works well with any number of inputs.

Product Pooling - outputs the product of inputs. Introduced by Stockfish, not common in machine learning in general. Only works well with 2 inputs. This one also appears to have similar benefits to sigmoid (quantmoid4); it increases the network's capacity, while other pooling layers only allow reducing dimensionality.

A simple input feature set.

For the purpose of illustration we will consider a simple set of inputs based on piece placement. We will call it "A" features, because they will represent "All pieces".

There are 64 squares on the board, 6 piece types (pawn, knight, bishop, rook, queen, king), and 2 colors (white, black). What we want to encode as inputs are the positions of pieces, so each input will correspond to some (square, piece_type, color) tuple. There are 64*6*2=768 such tuples. If there is a piece P of color C on the square S we set the input (S, P, C) to 1, otherwise, we set it to 0. Even though the total number of inputs is 768 there can only be 32 non-zero inputs in any given legal chess position, because there are only at most 32 pieces on the board. Moreover, any move can only change at most 4 inputs (castling), and the average should be below 3.

The binary and sparse nature of the inputs is utilized when passing the features to the neural network - the input is simply the list of features (indices), there's no need for a full input vector as other positions have value 0 and we know that each active feature has a value 1 associated with it.

Let's look at an example position 1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1.



On the board above we have 4 active features: (A1, king, white), (C3, pawn, white), (B8, king, black), (D4, rook, black).

Now let's consider the move c4 - the only feature that became invalid is the (C3, pawn, white), it needs to be replaced with (C4, pawn, white).

Now let's consider the move cxd4 - the pawn moved, so like before we remove (C3, pawn, white) and add (D4, pawn, white). But also the rook got removed from the board, so we have to remove (D4, rook, black) too. This is still less work than recreating the inputs from scratch!

A simple NNUE network

We will use our "A" feature set from the previous paragraph, so we have 768 inputs. The layers for the purpose of this illustration will be the 3 linear layers, 768->8, 8->8, 8->1. All layers are linear, and all hidden neurons use ClippedReLU activation function. The image below illustrates the architecture:



The flow is from the left to the right. The first layer is a large fully connected layer with 768 inputs, but only a small fraction of them is non-zero for each position - sparse matrix-vector multiplication can be utilized. Hidden layers are much smaller and always computed with dense matrix-vector multiplication. At the end, we get 1 output, which is usually trained to be the centipawn evaluation of the position (or proportional to it).

Consideration of networks size and cost.

Choosing the right architecture is tricky as it's an accuracy/performance trade-off. Large networks provide more accurate evaluation, but the speed impact might completely negate the gains in real play. For example Stockfish slowly transitioned from 256x2->32->32->1 to 1024x2->8->32->1.

Feature set

When choosing a feature set it might be tempting to go into complicated domain-specific knowledge, but the costs associated make simpler solutions more attractive. HalfKP, explained in detail later, is very simple, fast, and good enough. More sophisticated feature sets have been tried but they usually cannot combat the hit on performance. HalfKP features are easy to calculate, and change little from position to position.

Size also has to be considered. For the 256x2->32->32->1 architecture HalfKP inputs require about 10 million parameters in the first layer, which amounts to 20MB after quantization. For some users it might not be an issue to have a very large set of features, with possibly hundreds of millions of parameters, but for a typical user it's inconvenient. Moreover, increasing the feature set size may reduce the training speed for some implementations, and certainly will require more time to converge.

First set of hidden neurons

The number of outputs in the first layer is the most crucial parameter, and also has the highest impact on speed and size. The costs associated with this parameter are two-fold. For one, it increases the number of operations required when updating the accumulator. Second, for optimized implementations, one must consider the number of available registers - in Stockfish going past 256 neurons requires multiple passes over the feature indices as AVX2 doesn't have enough registers. It also partially determines the size of the first dense linear layer, which also greatly contributes to the total cost.

Further layers

Unlike in typical networks considered in machine learning here most of the knowledge is stored in the first layer, and because of that adding further small layers near the output adds little to accuracy, and may even be harmful if quantization is employed due to error accumulation. NNUE networks are kept unusually shallow, and keeping the size of the later layers small increases performance.

Accumulator

Even though we observed that few inputs change from position to position we have yet to take advantage of that. Recall that a linear layer is just adding some weight matrix columns together. Instead of recomputing the first set of hidden neurons for each position we can keep them as part of the position's state, and update it on each move based on what features (columns) were added or removed! We have to handle only two simple cases:

the feature i was removed from the input (1 -> 0) - subtract column i of the weight matrix from the accumulator

the feature i was added to the input (0 -> 1) - add column i of the weight matrix to the accumulator

For a single move, it's trivial to find which "A" features changed - we know what piece we're moving, from where, and where to. Captures and promotions can be considered as a piece disappearing or appearing from nowhere.

However, care must taken when using floating point values. Repeatedly adding and subtracting floats results in error that accumulates with each move. It requires careful evaluation of whether the error is small enough for the net to still produce good results. Thankfully, it is best implemented such that the accumulator is not updated when undoing a move. Instead, it is simply stored on the search stack, so the error is bounded by O(MAX_DEPTH) and can mostly be ignored.

When using quantization this is no longer a problem, the incremental implementation is consistent, but now there is a possibility of overflowing the accumulator (regardless of whether incremental updates are used or not). The quantization scheme must be chosen such that no combination of possible active features can exceed the maximum value.

HalfKP

HalfKP is the most common feature set and other successful ones build on top of it. It fits in a sweet spot of being just the right size, and requiring very few updates per move on average. Each feature is a tuple (our_king_square, piece_square, piece_type, piece_color), where piece_type is not a king (in HalfKA feature set kings are included). This means that for each king position there is a set of features P, which are (piece_square, piece_type, piece_color). This allows the net to better understand the pieces in relation to the king. The total number of features is 64*64*5*2=40960. (Note that there is a leftover from Shogi in the current Stockfish implementation and there are 64 additional features that are unused, but we will disregard them in this document). The feature index can be calculated as

p_idx = piece_type * 2 + piece_colorhalfkp_idx = piece_square + (p_idx + king_square * 10) * 64

The one special case that needs to be handled is when the king moves, because it is tied to all the features. All features are changed, so an accumulator refresh is executed. This makes king moves more costly but on average it still keeps the number of updates per evaluation low.

Now, you might ask, "but which king?!". The answer is both...

Multiple perspectives, multiple accumulators

This is where we need to start accounting for the features of both sides separately. The white side will keep its own accumulator, and the black side its own accumulator too. Effectively, it means that the maximum active number of features is twice as high as for a simple feature set with only one perspective. There will be twice as many updates and the accumulator will be twice as large in total, but overall this tradeoff between speed and accuracy is worth it. This approach inevitably creates some problems, options, and choices with regard to the exact model topology. Let's go through them one by one.

How to combine multiple accumulator perspectives?



Since we now have two accumulators, we need to somehow combine them into one vector that gets passed further into the network. This can be solved in two (three) ways. Let's denote the accumulator for white as A_w, and the accumulator for black as A_b.

concatenate the A_w and A_b, placing A_w first and A_b second. This is the simplest option. The output in this case is always relative to the white's perspective.

concatenate the A_w and A_b, placing A_w first if it's white to move, otherwise A_b first, and the other accumulator second. This approach has the advantage that the net can learn tempo. It now knows whose turn it is, which is an important factor in chess and can have a huge impact on evaluation of some positions. The output in this case is always relative to the side to move perspective.

Either 1 or 2, but instead of concatenating interleave. So A_w[0], A_b[0], A_w[1], A_b[1], .... This might be advantageous in some exotic architectures where not always the whole combined accumulator is used, in which case interleaving means that the slice used always contains the same number of outputs from white's and from black's perspectives. This might become useful, for example when employing structured sparsity to the first hidden layer, which ultimately works on the subset of the accumulator.

Which set of weights to use for each perspective?



So we compute the features for white and black the same, are their weights related? They can be, but it's not required. Engines differ in the handling of this.

Same weights for both perspectives. This means the board state needs to somehow be oriented. Otherwise white king on E1 would produce a different subset of features than a black king on E8, and white king on G4 would produce the same subset of features as a black king on G4. That's bad. The solution is to mirror the position and swap the color of the pieces for black's perspective; then the piece placement to feature mapping is logical for both. White king on E1 from white's perspective should be the same as a black king on E8 from black's perspective. Now you may think that flip is the way to go, but while chess has vertical symmetry, Shogi has rotational symmetry. The initial implementation of HalfKP in Stockfish used rotation to change the perspective, which is arguably incorrect for chess, but it worked surprisingly well.

Different weights for different perspectives. Is the white king on E1 actually equal to black king on E8? What about other pieces? Arguably one plays the game differently as black compared to as white, and it seems it makes sense to use different features for these perspectives. This is how some engines do it, and there's nothing wrong with this. The only downsides are larger size and slightly longer training time, but other than that it might even be better! It also completely removes the discussion about flip or rotate, which makes the implementation simpler.

HalfKP example and network diagram

Similar to the diagram above for the "A" feature set, here is the diagram for the same network but with HalfKP feature set, with combined weights. With a change that both accumulators are of size 4, so the network is in the end HalfKP[40960]->4x2->8->1

Let's look at the same example position as before: 1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1.



Now we have two perspectives, and will list the features for both of them separately. Remember the features are (our_king_square, piece_square, piece_type, piece_color) and we use flip to orient the squares for black and the colors are reversed! (One can think of the "color" as "us" or "them")

White's perspective: (A1, C3, pawn, white), (A1, D4, rook, black)

Blacks's perspective: (B1, C6, pawn, black), (B1, D5, rook, white)

The network diagram looks more interesting now.



Forward pass implementation

In this part, we will look at model inference as it could be implemented in a simple chess engine. We will work with floating point values for simplicity here. Input generation is outside of the scope of this implementation.

Example network

We will take a more generally defined network, with architecture FeatureSet[N]->M*2->K->1. The layers will therefore be:

L_0: Linear N->M

C_0: Clipped ReLU of size M*2

L_1: Linear M*2->K

C_1: Clipped ReLU of size K

L_2: Linear K->1

Layer parameters

Linear layers have 2 parameters - weights and biases. We will refer to them as L_0.weight and L_0.bias respectively. The layers also contain the number of inputs and outputs, in L_0.num_inputs and L_0.num_outputs respectively.

Here something important has to be said about the layout of the weight matrix. For sparse multiplication, the column-major (a column is contiguous in memory) layout is favorable, as we're adding columns, but for dense multiplication this is not so clear and a row-major layout may be preferable. For now we will stick to the column-major layout, but we may revisit the row-major one when it comes to quantization and optimization. For now, we assume L_0.weight allows access to the individual elements in the following form: L_0.weight[column_index][row_index].

The code presented is very close to C++ but technicalities might be omitted.

Accumulator

The accumulator can be represented by an array that is stored along other position state information on the search stack.

struct NnueAccumulator { // Two vectors of size M. v[0] for white's, and v[1] for black's perspectives. float v[2][M]; // This will be utilised in later code snippets to make the access less verbose float* operator[](Color perspective) { return v[perspective]; }};

The accumulator can either be updated lazily on evaluation, or on each move. It doesn't matter here, but it has to be updated somehow. Whether it's better to update lazily or eagerly depends on the number of evaluations done during search. For updates, there are two cases, as laid out before:

The accumulator has to be recomputed from scratch.

The previous accumulator is reused and just updated with changed features

Refreshing the accumulator

void refresh_accumulator( const LinearLayer& layer, // this will always be L_0 NnueAccumulator& new_acc, // storage for the result const std::vector<int>& active_features, // the indices of features that are active for this position Color perspective // the perspective to refresh) { // First we copy the layer bias, that's our starting point for (int i = 0; i < M; ++i) { new_acc[perspective][i] = layer.bias[i]; } // Then we just accumulate all the columns for the active features. That's what accumulators do! for (int a : active_features) { for (int i = 0; i < M; ++i) { new_acc[perspective][i] += layer.weight[a][i]; } }}

Updating the accumulator

void update_accumulator( const LinearLayer& layer, // this will always be L_0 NnueAccumulator& new_acc, // it's nice to have already provided storage for // the new accumulator. Relevant parts will be overwritten const NNueAccumulator& prev_acc, // the previous accumulator, the one we're reusing const std::vector<int>& removed_features, // the indices of features that were removed const std::vector<int>& added_features, // the indices of features that were added Color perspective // the perspective to update, remember we have two, // they have separate feature lists, and it even may happen // that one is updated while the other needs a full refresh) { // First we copy the previous values, that's our starting point for (int i = 0; i < M; ++i) { new_acc[perspective][i] = prev_acc[perspective][i]; } // Then we subtract the weights of the removed features for (int r : removed_features) { for (int i = 0; i < M; ++i) { // Just subtract r-th column new_acc[perspective][i] -= layer.weight[r][i]; } } // Similar for the added features, but add instead of subtracting for (int a : added_features) { for (int i = 0; i < M; ++i) { new_acc[perspective][i] += layer.weight[a][i]; } }}

And that's it! Pretty simple, isn't it?

Linear layer

This is simple vector-matrix multiplication, what could be complicated about it you ask? Nothing for now, but it will get complicated once optimization starts. Right now we won't optimize, but we will at least write a version that uses the fact that the weight matrix has a column-major layout.

float* linear( const LinearLayer& layer, // the layer to use. We have two: L_1, L_2 float* output, // the already allocated storage for the result const float* input // the input, which is the output of the previous ClippedReLU layer) { // First copy the biases to the output. We will be adding columns on top of it. for (int i = 0; i < layer.num_outputs; ++i) { output[i] = layer.bias[i]; } // Remember that rainbowy diagram long time ago? This is it. // We're adding columns one by one, scaled by the input values. for (int i = 0; i < layer.num_inputs; ++i) { for (int j = 0; j < layer.num_outputs; ++j) { output[j] += input[i] * layer.weight[i][j]; } } // Let the caller know where the used buffer ends. return output + layer.num_outputs;}

ClippedReLU

float* crelu(, int size, // no need to have any layer structure, we just need the number of elements float* output, // the already allocated storage for the result const float* input // the input, which is the output of the previous linear layer) { for (int i = 0; i < size; ++i) { output[i] = min(max(input[i], 0), 1); } return output + size;}

Putting it together

In a crude pseudo code. The feature index generation is left as an exercise for the reader.

void Position::do_move(...) { ... // do the movey stuff for (Color perspective : { WHITE, BLACK }) { if (needs_refresh[perspective]) { refresh_accumulator( L_0, this->accumulator, this->get_active_features(perspective), perspective ); } else { update_accumulator( L_0, this->accumulator, this->get_previous_position()->accumulator, this->get_removed_features(perspective), this->get_added_features(perspective), perspective ); } }}float nnue_evaluate(const Position& pos) { float buffer[...]; // allocate enough space for the results // We need to prepare the input first! We will put the accumulator for // the side to move first, and the other second. float input[2*M]; Color stm = pos.side_to_move; for (int i = 0; i < M; ++i) { input[ i] = pos.accumulator[ stm][i]; input[M+i] = pos.accumulator[!stm][i]; } float* curr_output = buffer; float* curr_input = input; float* next_output; // Evaluate one layer and move both input and output forward. // The last output becomes the next input. next_output = crelu(2 * L_0.num_outputs, curr_output, curr_input); curr_input = curr_output; curr_output = next_output; next_output = linear(L_1, curr_output, curr_input); curr_input = curr_output; curr_output = next_output; next_output = crelu(L_1.num_outputs, curr_output, curr_input); curr_input = curr_output; curr_output = next_output; next_output = linear(L_2, curr_output, curr_input); // We're done. The last layer should have put 1 value out under *curr_output. return *curr_output;}

And that's it! That's the whole network. What do you mean you can't use it?! OH RIGHT, you don't have a net trained, what a bummer.

Training a net with pytorch

This will be very brief, as this is on the nnue-pytorch repo after all so you can just look up the code! We will not explain how pytorch works, but we will, however, explain some of the basics, and the quirks needed to accommodate this exotic use case.

Let's continue using the architecture from the forward pass implementation.

Model specification

Pytorch has built-in types for linear layers, so defining the model is pretty simple.

class NNUE(nn.Module): def __init__(self): super(NNUE, self).__init__() self.ft = nn.Linear(NUM_FEATURES, M) self.l1 = nn.Linear(2 * M, N) self.l2 = nn.Linear(N, K) # The inputs are a whole batch! # `stm` indicates whether white is the side to move. 1 = true, 0 = false. def forward(self, white_features, black_features, stm): w = self.ft(white_features) # white's perspective b = self.ft(black_features) # black's perspective # Remember that we order the accumulators for 2 perspectives based on who is to move. # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors. accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1)) # Run the linear layers and use clamp_ as ClippedReLU l1_x = torch.clamp(accumulator, 0.0, 1.0) l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0) return self.l2(l2_x)

Thankfully, Pytorch handles backpropagation automatically through automatic differentiation. Neat! The hard bit now is, maybe surprisingly, feeding the data.




Show thinking
The provided details elaborate on the core principles and architecture of Efficiently Updatable Neural Networks (NNUE), particularly as applied in chess engines like Stockfish.

Here's a re-iteration of the key aspects:

What is NNUE?
NNUE (Efficiently Updatable Neural Network) is a neural network architecture designed for board games like chess and Shogi, optimizing for scenarios where network inputs change minimally between successive evaluations. It was initially developed for Shogi in 2018 and later integrated into Stockfish for chess.

Core Principles:

Sparsity of Inputs: The network is designed to handle inputs where only a small fraction of values are non-zero (e.g., 0.1% for current best architectures). This allows the first, often largest, layer to be evaluated very quickly by only processing the relevant (non-zero) inputs.

Incremental Updates: This is a crucial optimization. Instead of re-evaluating the entire network (or its most computationally expensive part, the first layer) from scratch for every position, NNUE leverages the small changes between board states (e.g., a single move). It updates an "accumulator" that represents the output of the first layer by simply adding or subtracting weights corresponding to the features that changed. This significantly speeds up evaluation.

Low-Precision Integer Inference (Quantization): NNUE networks are specifically designed for fast evaluation using low-precision integer arithmetic (int8/int16). This provides maximum performance on modern CPUs, sacrificing negligible accuracy for substantial speed gains compared to floating-point calculations.

Useful Layers in NNUE
NNUE primarily relies on simple layers that can be efficiently implemented in a low-precision integer domain:

Linear (Fully Connected) Layer: Performs a matrix multiplication (y = Ax + b). Its efficiency comes from the sparse input principle; only columns of the weight matrix corresponding to non-zero inputs need to be processed.

Clipped ReLU Layer: An activation function defined as y = min(max(x, 0), 1). It introduces non-linearity while keeping the output values within a bounded range, which is important for aggressive quantization.

Quantmoid4: An approximation of the Sigmoid function designed specifically for quantized integer implementations, outputting values in the range [0, 126]. This is used when a smooth, saturating activation is desired but the computational cost of a true sigmoid is prohibitive.

Network Structure and Feature Set
Shallow Networks: NNUE models are typically shallow (2-4 layers). Most of the "knowledge" is embedded in the large first layer, which benefits most from input sparsity. Subsequent layers are kept small to maintain performance.

HalfKP Feature Set: This is a widely used and efficient feature set. Each feature is a tuple (our_king_square, piece_square, piece_type, piece_color). The total number of features is 64 * 64 * 5 * 2 = 40960.

Feature Index Calculation: p_idx = piece_type * 2 + piece_color; halfkp_idx = piece_square + (p_idx + king_square * 10) * 64.

King Moves: If the king moves, all features tied to it change, necessitating a full "refresh" of the accumulator for that perspective.

Multiple Perspectives (Accumulators): NNUE utilizes two separate accumulators, one for white's perspective and one for black's. These are combined into a single vector for subsequent layers, often with the side-to-move's accumulator placed first to allow the network to learn "tempo."

The Accumulator and Incremental Updates
The accumulator is a critical component. Instead of recomputing the first layer's output for every position, its state is stored and updated incrementally.

refresh_accumulator: Used when a full re-computation is needed (e.g., after a king move), it sums the biases and the weights of all active features.

update_accumulator: Used for typical moves. It starts with the previous position's accumulator, then subtracts the weights of features that were removed and adds the weights of features that were added. This significantly reduces computation.

Dart Implementation with Your Deep Learning Files
Based on your provided Value, ValueVector, Neuron, Layer, and Module classes, here's how to conceptualize and implement the NNUE architecture in Dart, incorporating the details described above.

Note on Quantization: The provided Value class uses double (floating-point). While the NNUE concept emphasizes integer quantization for peak performance, this implementation will use double for the core calculations to align with your existing framework. For a production-grade NNUE, you would need to introduce fixed-point arithmetic or specialized integer types.

Let's define the necessary classes and functions.

Dart

import 'dart:math' as math;
import 'module.dart';
import 'layer.dart';
import 'neuron.dart';
import 'value.dart';
import 'value_vector.dart';

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
extension ValueClamp on Value {
  Value clamp(Value minVal, Value maxVal) {
    final clamped = Value(data.clamp(minVal.data, maxVal.data).toDouble(), {this, minVal, maxVal}, 'clamp');
    clamped._backward = () {
      if (data >= minVal.data && data <= maxVal.data) {
        grad += clamped.grad;
      }
    };
    return clamped;
  }
}

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
            (_) => List.generate(numOutputsPerPerspective,
                (_) => Value(math.Random().nextDouble() * 0.01))), // Small random weights
        biases = List.generate(numOutputsPerPerspective, (_) => Value(0.0)); // Zero biases

  // Refreshes the accumulator for a given perspective
  // activeFeatures: indices of currently active features for this perspective
  void refreshAccumulator(
      List<int> activeFeatures, List<Value> accumulatorPerspective) {
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
      List<int> addedFeatures) {
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
            2, (_) => List.generate(numOutputsPerPerspective, (_) => Value(0.0)));

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
    int currentKingSq = (perspective == WHITE) ? whiteKingSquare : blackKingSquare;

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
  List<int> getAddedFeatures(int perspective) => [];   // Dummy

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
    required int hiddenLayer2Size, // N in K->N (note: text uses N for L2 input, K for L1 input)
  })  : ft = NnueFirstLayerWeights(numFeatures, numOutputsPerPerspective),
        hiddenLayer1 = Layer.fromNeurons(2 * numOutputsPerPerspective, hiddenLayer1Size), // M*2 inputs
        hiddenLayer2 = Layer.fromNeurons(hiddenLayer1Size, hiddenLayer2Size), // K inputs
        outputLayer = Layer.fromNeurons(hiddenLayer2Size, 1), // N inputs, 1 output
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
          currentAccumulator[BoardState.WHITE]);
    } else {
      ft.updateAccumulator(
          currentAccumulator[BoardState.WHITE],
          currentAccumulator[BoardState.WHITE], // Using same for prev_acc in dummy
          currentBoard.getRemovedFeatures(BoardState.WHITE),
          currentBoard.getAddedFeatures(BoardState.WHITE));
    }

    // Black's perspective
    if (currentBoard.needsRefresh(BoardState.BLACK)) {
      ft.refreshAccumulator(
          currentBoard.getActiveFeatures(BoardState.BLACK),
          currentAccumulator[BoardState.BLACK]);
    } else {
      ft.updateAccumulator(
          currentAccumulator[BoardState.BLACK],
          currentAccumulator[BoardState.BLACK], // Using same for prev_acc in dummy
          currentBoard.getRemovedFeatures(BoardState.BLACK),
          currentBoard.getAddedFeatures(BoardState.BLACK));
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
        combinedAccumulator.values.map((v) => clippedReLU(v)).toList());

    // 4. Linear Layer 1 (L_1)
    ValueVector l1Output = hiddenLayer1.forward(l1Input);

    // 5. Clipped ReLU on L_1 output (input to L_2)
    ValueVector l2Input =
        ValueVector(l1Output.values.map((v) => clippedReLU(v)).toList());

    // 6. Linear Layer 2 (L_2)
    ValueVector l2Output = hiddenLayer2.forward(l2Input);

    // 7. Clipped ReLU on L_2 output (input to final output layer)
    ValueVector finalInputToOutput =
        ValueVector(l2Output.values.map((v) => clippedReLU(v)).toList());

    // 8. Output Layer (L_3)
    Value finalEvaluation = outputLayer.forward(finalInputToOutput).values.first;

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
      20: {'type': BoardState.KING, 'color': BoardState.WHITE}, // White King on square 20
      45: {'type': BoardState.KING, 'color': BoardState.BLACK}, // Black King on square 45
    },
    whiteKingSquare: 20,
    blackKingSquare: 45,
    sideToMove: BoardState.WHITE,
  );

  // The accumulator will be managed on the "search stack"
  final NnueAccumulator currentPositionAccumulator =
      NnueAccumulator(numOutputsPerPerspective);

  // Perform a forward pass (initial evaluation)
  final nnueEvaluation =
      nnueModel.forward(initialBoard, currentPositionAccumulator);
  print(
      "Initial NNUE Evaluation: ${nnueEvaluation.data.toStringAsFixed(4)}");

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
  final nnueEvaluationAfterUpdate =
      nnueModel.forward(nextBoardState, currentPositionAccumulator);
  Value dummyLossAfterUpdate =
      (targetEvaluation - nnueEvaluationAfterUpdate).pow(2);

  print(
      "NNUE Evaluation After 1 Move & Update: ${nnueEvaluationAfterUpdate.data.toStringAsFixed(4)}");
  print("Dummy Loss After 1 Move & Update: ${dummyLossAfterUpdate.data.toStringAsFixed(4)}");
  print("\nThis demonstrates the conceptual flow of NNUE with incremental updates and training.");
  print("A complete implementation would require robust board representation, feature generation, and accumulator management on a search stack.");
}






Feature factorization

Let's focus on the features again. We will take a closer look at the HalfKP feature set. Recall, that HalfKP features are indexed by tuples of form (king_square, piece_square, piece_type, piece_color), where piece_type != KING.

The HalfKP feature set was formed by specialization of the P feature set for every single king square on the board. This in turn increased the feature set size, and caused the accesses to become much more sparse. This sparsity directly impacts how much each feature is seen during training, and that negatively impacts the learning of weights.

Feature factorization effectively, and efficiently, relates features together during training, so that more features are affected during each step of training. This is particularly important during early stages of training, because it results in even the rarest of feature weights being populated quickly with reasonable values.

Feature factorization works by introducing a "virtual" feature set (as opposed to the "real" feature set, here HalfKP) that contains denser features, each being directly related to (and, importantly, redundant with) one or more "real" features. These "virtual" features are present only during the training process, and will learn the common factor for all "real" features they relate to. Let's see how it works in case of HalfKP.

HalfKP is just P taken 64 times, once for each king square, as mentioned previously. Each P feature is therefore related to 64 HalfKP features, and will learn the common factor for a (piece_square, piece_type, piece_color) feature for all possible king positions.

Because "virtual" features are redundant with the "real" features their weights can be coalesced into the "real" features weights after the training is finished. The way to coalesce them follows from the computation performed in the network layer (the feature transformer).

Virtual feature coalescing

So how can we coalesce them? Let's look at how matrix and vector multiplication is done again. Consider the example position from before (1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1).

:

Let's focus on the feature (A1, C3, pawn, white). Now, we're also gonna add the corresponding P feature (C3, pawn, white). What happens when the input goes through the first layer?

accumulator += weights[(A1, C3, pawn, white)];accumulator += weights[(C3, pawn, white)];

which is equivalent to

accumulator += weights[(A1, C3, pawn, white)] + weights[(C3, pawn, white)];

So the relation is very simple. We just need to add the weights of each P feature to all the related HalfKP feature weights!

Other factors

Sometimes it's possible to add even more factors. It should be noted, however, that just adding more factors doesn't necessarily improve the training and may even cause it to regress. In general, whether using some factors helps or not depends on the training setup and the net being trained. It's always good to experiment with this stuff. With that said, however, we can consider for example the following factors for HalfKP.

"K" factors



The king position, 64 features. This one requires some careful handling as a single position has this feature multiple times - equal to the number of pieces on the board. This virtual feature set is needed purely because with HalfKP the king position feature is not encoded anywhere. HalfKA doesn't need it for example because it specifically has the feature for the king's position. In general, handling this is tricky, it may even require reducing the gradient for these features (otherwise the gradient is input*weight, but input is large compared to others).

"HalfRelativeKP" factors



In HalfKP we use the absolute piece position, but what if we encoded the position as relative to the king? There are 15x15 such relative positions possible, and most of them correspond 1:many to some HalfKP feature. The HalfRelativeKP feature index could be calculated for example like this:

int get_half_relative_kp_index(Color perspective, Square king_sq, Square piece_sq, Piece piece){ const int p_idx = static_cast<int>(piece.type()) * 2 + (piece.color() != perspective); const Square oriented_king_sq = orient_flip(perspective, king_sq); const Square oriented_piece_sq = orient_flip(perspective, piece_sq); // The file/rank difference is always in range -7..7, and we need to map it to 0..15 const int relative_file = oriented_piece_sq.file() - oriented_king_sq.file() + 7; const int relative_rank = oriented_piece_sq.rank() - oriented_king_sq.rank() + 7; return (p_idx * 15 * 15) + (relative_file * 15) + relative_rank;}

Real effect of the factorizer

While the factorizer helps the net to generalize, it seems to only be relevant in the early stages, that is when the net doesn't really know anything yet. It accelerates the early stages of training and reduces the sparsity of the input (some inputs are very rare otherwise). But it quickly becomes unimportant and in later stages of the training can be removed to gain some training speed (after all it can add a lot of active features).

Loss functions and how to apply them

The Goal

Training a network is really just minimizing a loss function, which needs to be smooth and have a minimum at the "optimal" evaluation (the training target). For the purpose of NNUE, this is done by gradient descent through usual machine learning methods (there are also non-gradient methods that are not described here).

Converting the evaluation from CP-space to WDL-space

By CP-space we mean the centipawn scale (or something proportional, like engine's internal units). By WDL-space we mean 0=loss, 0.5=draw, 1=win.

It's of course possible to apply the loss function directly on the evaluation value (in CP-space), but this can lead to large gradients (or a lot of hyperparameter tuning), restricts the set of loss functions available, and doesn't allow using results for loss. We will focus on evaluation in WDL-space. But how to convert between these spaces? Usually, the evaluation to performance correspondence can be well-fitted by a sigmoid. For example, in some data generated by Stockfish we have:



so in the code we may do the following:

scaling_factor = 410 # this depends on the engine, and maybe even on the datawdl_space_eval = torch.sigmoid(cp_space_eval / scaling_factor)

This transformation also has the nice effect that large evaluations become "closer" together, which aligns well with the real play, where large evaluations don't need to be that precise.

Using results along the evaluation

With the values for which we will compute loss being in WDL-space, we may now interpolate them with game results. We will introduce a lambda_ parameter that governs the interpolation.

# game_result is in WDL-spacewdl_value = lambda_ * wdl_space_eval + (1 - lambda_) * game_result

The interpolation can also be applied to the loss.

loss_eval = ... # loss between model eval and position evalloss_result = ... # loss between model eval and game resultloss = lambda_ * loss_eval + (1 - lambda_) * loss_result

Which way works better depends on your case 😃

Mean Squared Error (MSE)

Now we know what we're trying to fit; let's look at how we will fit them.

This is a very simple loss function that just takes a square of the difference between the predicted value and the target. This results in a nice linear gradient.

With interpolation applied before:

scaling = ... # depends on the engine and data. Determines the shape of # the sigmoid that transforms the evaluation to WDL space # Stockfish uses values around 400wdl_eval_model = sigmoid(model(...) / scaling)wdl_eval_target = sigmoid(target / scaling)wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_resultloss = (wdl_eval_model - wdl_value_target)**2

With interpolation applied after:

scaling = ...wdl_eval_model = sigmoid(model(...) / scaling)wdl_eval_target = sigmoid(target / scaling)loss_eval = (wdl_eval_model - wdl_eval_target)**2loss_result = (wdl_eval_model - game_result)**2loss = lambda_ * loss_eval + (1 - lambda_) * loss_result

Note: in practice, the exponent can be >2. Higher exponents give more weight towards precision at a cost of accuracy. Stockfish networks had good training results with an exponent of 2.6 for example.

loss





grad





Cross entropy

This loss function is usually used for continuous classification problems, and our use case could be considered one.

Care must be taken around domain boundaries. Usually, a very small value (epsilon) is added such that the values never reach 0 under the logarithm.

With interpolation applied before:

epsilon = 1e-12 # to prevent log(0)scaling = ...wdl_eval_model = sigmoid(model(...) / scaling)wdl_eval_target = sigmoid(target / scaling)wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result# The first term in the loss has 0 gradient, because we always# differentiate with respect to `wdl_eval_model`, but it makes the loss nice# in the sense that 0 is the minimum.loss = (wdl_value_target * log(wdl_value_target + epsilon) + (1 - wdl_value_target) * log(1 - wdl_value_target + epsilon)) -(wdl_value_target * log(wdl_eval_model + epsilon) + (1 - wdl_value_target) * log(1 - wdl_eval_model + epsilon))

With interpolation applied after:

epsilon = 1e-12 # to prevent log(0)scaling = ...wdl_eval_model = sigmoid(model(...) / scaling)wdl_eval_target = sigmoid(target / scaling)# The first term in the loss has 0 gradient, because we always# differentiate with respect to `wdl_eval_model`, but it makes the loss nice# in the sense that 0 is the minimum.loss_eval = (wdl_eval_target * log(wdl_eval_target + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_target + epsilon)) -(wdl_eval_target * log(wdl_eval_model + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_model + epsilon))loss_result = (game_result * log(wdl_eval_target + epsilon) + (1 - game_result) * log(1 - wdl_eval_target + epsilon)) -(game_result * log(wdl_eval_model + epsilon) + (1 - game_result) * log(1 - wdl_eval_model + epsilon))loss = lambda_ * loss_eval + (1 - lambda_) * loss_result

loss





grad





Qua



The new details delve into advanced aspects of NNUE, particularly feature factorization for improved training and various loss functions used in the training process.

Feature Factorization
Feature factorization is a technique primarily used during NNUE training to address the sparsity of inputs in feature sets like HalfKP. Because HalfKP features are very specific (e.g., (king_square, piece_square, piece_type, piece_color)), many feature weights might be rarely updated, leading to slow learning.

How it Works:

Virtual Features: During training, "virtual" feature sets are introduced alongside the "real" HalfKP features. These virtual features are denser and represent common factors across multiple HalfKP features.

Redundancy and Coalescing: The virtual features are designed to be redundant. For example, a P feature (piece_square, piece_type, piece_color) would be active whenever any of the 64 HalfKP features (one for each king square) that share that P feature are active.

Training Benefit: By activating these denser virtual features, more weights (both "real" and "virtual") are updated during each training step. This accelerates the learning process, especially in early stages, by quickly populating weights for rare features with reasonable values.

Post-Training Coalescing: After training, the weights of the virtual features are "coalesced" into the weights of the real features. This means adding the learned weights of a virtual feature to all the real features it corresponds to. This simplifies the trained model for inference, as the virtual features are no longer needed. For HalfKP and P features, this simply means:
coalesced_HalfKP_weights[H] = trained_HalfKP_weights[H] + trained_P_weights[P_corresponding_to_H]
where H is a HalfKP feature and P is its corresponding P feature.

Other Factors:

"K" Factors (King position): Can be added to account for the king's absolute position, which HalfKP doesn't explicitly encode. This needs careful handling due to its high activation frequency (equal to the number of pieces).

"HalfRelativeKP" Factors: Encode piece positions relative to the king. This can provide additional generalization.

Real Effect: Factorization primarily helps in the early stages of training by reducing input sparsity and accelerating learning. It can be removed in later stages to improve training speed if it no longer provides significant benefit.

Loss Functions for NNUE Training
The goal of training is to minimize a loss function using gradient descent. NNUE commonly transforms the evaluation from "centipawn (CP) space" (engine's internal evaluation units) to "Win-Draw-Loss (WDL) space" (0=loss, 0.5=draw, 1=win) using a sigmoid function:

wdl_space_eval = sigmoid(cp_space_eval / scaling_factor)

This transformation is beneficial because:

It aligns with the real-world outcome (win/loss/draw).

It compresses large evaluations, making them less sensitive to small changes, which mirrors how large evaluations behave in practical play.

Interpolating with Game Results:
A lambda_ parameter is introduced to blend the model's WDL evaluation with the actual game result (if available in the training data):

wdl_value = lambda_ * wdl_space_eval + (1 - lambda_) * game_result

This interpolation can be applied to the target before calculating the loss, or to the loss components themselves.

Common Loss Functions:

Mean Squared Error (MSE):

Formula: loss = (predicted_value - target_value)^2

This results in a linear gradient.

The exponent can be >2 (e.g., 2.6 in Stockfish) to give more weight to precision.

Cross Entropy:

Typically used for classification, but can be adapted for continuous values like WDL.

Requires adding a small epsilon to prevent log(0).

The formula involves terms like target * log(prediction) + (1 - target) * log(1 - prediction).

Both MSE and Cross-Entropy can be used, with the choice often depending on empirical results for a specific training setup.

Dart Implementation with Feature Factorization and Loss Functions
To incorporate feature factorization, we primarily need to adjust the NnueFirstLayerWeights and the BoardState (or feature generation) for training. For loss functions, we'll implement the sigmoid transformation and the MSE/Cross-Entropy losses.

Refinements and New Components:

Value.sigmoid(scalingFactor): Add a sigmoid function to the Value class or as an extension.

NnueFirstLayerWeights (ft): This class already holds the L_0 weights. For training with factorization, we'd conceptually have additional "virtual" weights that are added during the forward pass of training. After training, these virtual weights would be added to the ft.weights directly. For a live inference model, we only need the coalesced weights, so the current NnueFirstLayerWeights structure is appropriate for the final model. The factorization itself is part of the training data preparation and the training loop's forward pass.

BoardState (or FeatureGenerator): This would be responsible for generating both the HalfKP features and any "virtual" features (like P features) during training. For inference, it only generates HalfKP.

Let's focus on integrating the sigmoid conversion and MSE loss into the main training loop, as implementing the full factorization and Quantmoid4 would require a much more extensive training setup (data loaders, separate training code, etc.) beyond the scope of providing a single model file. The current model structure correctly represents the inference part of an NNUE that has already been trained and coalesced.

Dart

import 'dart:math' as math;
import 'module.dart';
import 'layer.dart';
import 'neuron.dart';
import 'value.dart';
import 'value_vector.dart';

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
extension ValueClamp on Value {
  Value clamp(Value minVal, Value maxVal) {
    final clamped = Value(data.clamp(minVal.data, maxVal.data).toDouble(), {this, minVal, maxVal}, 'clamp');
    clamped._backward = () {
      if (data >= minVal.data && data <= maxVal.data) {
        grad += clamped.grad;
      }
    };
    return clamped;
  }

  // Add sigmoid function to Value for WDL conversion
  // y = 1 / (1 + exp(-x))
  Value sigmoid() {
    final expNegX = (-this).exp(); // Using Value.exp() if implemented
    final out = Value(1.0 / (1.0 + expNegX.data), {this, expNegX}, 'sigmoid');
    out._backward = () {
      // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
      // out.data is already sigmoid(this.data)
      grad += out.grad * out.data * (1.0 - out.data);
    };
    return out;
  }
}

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
            (_) => List.generate(numOutputsPerPerspective,
                (_) => Value(math.Random().nextDouble() * 0.01))), // Small random weights
        biases = List.generate(numOutputsPerPerspective, (_) => Value(0.0)); // Zero biases

  // Refreshes the accumulator for a given perspective
  // activeFeatures: indices of currently active features for this perspective
  void refreshAccumulator(
      List<int> activeFeatures, List<Value> accumulatorPerspective) {
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
      List<int> addedFeatures) {
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
            2, (_) => List.generate(numOutputsPerPerspective, (_) => Value(0.0)));

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
        _cachedAddedFeaturesWhite = [2000];  // Dummy feature index
      } else {
        _cachedRemovedFeaturesBlack = [3000]; // Dummy feature index
        _cachedAddedFeaturesBlack = [4000];  // Dummy feature index
      }
    }
  }

  // Simulates getting active features for a given perspective (for refresh)
  // In a real engine, this would iterate through pieces and their squares.
  List<int> getActiveFeatures(int perspective) {
    final List<int> features = [];
    int currentKingSq = (perspective == WHITE) ? whiteKingSquare : blackKingSquare;

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
      final p_idx = pieceType * 2 + pieceColor; // Assuming pieceColor is 0 for white, 1 for black
      final halfkp_idx = square + (p_idx + currentKingSq * 10) * 64;

      // Ensure feature index is within expected bounds if any are passed
      if (halfkp_idx >= 0 && halfkp_idx < 40960) { // Max HalfKP features is 40960
         features.add(halfkp_idx);
      }
    });

    // Add some random features to make it slightly more dynamic for testing
    // In a real scenario, this is deterministic based on board.
    final random = math.Random();
    for (int i = 0; i < 5; i++) { // Add 5 random active features
      features.add(random.nextInt(40960));
    }

    return features.toSet().toList(); // Ensure unique features
  }

  // Determines if a full accumulator refresh is needed (e.g., king moved)
  bool needsRefresh(int perspective) {
    if (_previousState == null) return true; // Always refresh for the very first position

    int prevKingSq = (perspective == WHITE) ? _previousState!.whiteKingSquare : _previousState!.blackKingSquare;
    int currKingSq = (perspective == WHITE) ? whiteKingSquare : blackKingSquare;

    return prevKingSq != currKingSq; // Refresh if king moved
  }

  List<int> getRemovedFeatures(int perspective) {
    return (perspective == WHITE) ? _cachedRemovedFeaturesWhite : _cachedRemovedFeaturesBlack;
  }
  List<int> getAddedFeatures(int perspective) {
    return (perspective == WHITE) ? _cachedAddedFeaturesWhite : _cachedAddedFeaturesBlack;
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
    if (move == 'e4') { // White pawn move
      newPieces.removeWhere((sq, piece) => sq == 10 && piece['type'] == PAWN);
      newPieces[18] = {'type': PAWN, 'color': WHITE}; // New square for pawn
    } else if (move == 'Ke2') { // White king move, forces refresh for white
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
  })  : ft = NnueFirstLayerWeights(numFeatures, numOutputsPerPerspective),
        hiddenLayer1 = Layer.fromNeurons(2 * numOutputsPerPerspective, hiddenLayer1Size), // M*2 inputs
        hiddenLayer2 = Layer.fromNeurons(hiddenLayer1Size, hiddenLayer2Size), // K inputs
        outputLayer = Layer.fromNeurons(hiddenLayer2Size, 1), // N inputs, 1 output
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
          currentAccumulator[BoardState.WHITE]);
    } else {
      ft.updateAccumulator(
          currentAccumulator[BoardState.WHITE],
          currentAccumulator[BoardState.WHITE], // Assuming current is also previous for update path
          currentBoard.getRemovedFeatures(BoardState.WHITE),
          currentBoard.getAddedFeatures(BoardState.WHITE));
    }

    // Black's perspective
    if (currentBoard.needsRefresh(BoardState.BLACK)) {
      ft.refreshAccumulator(
          currentBoard.getActiveFeatures(BoardState.BLACK),
          currentAccumulator[BoardState.BLACK]);
    } else {
      ft.updateAccumulator(
          currentAccumulator[BoardState.BLACK],
          currentAccumulator[BoardState.BLACK], // Assuming current is also previous for update path
          currentBoard.getRemovedFeatures(BoardState.BLACK),
          currentBoard.getAddedFeatures(BoardState.BLACK));
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
        combinedAccumulator.values.map((v) => clippedReLU(v)).toList());

    // 4. Linear Layer 1 (L_1)
    ValueVector l1Output = hiddenLayer1.forward(l1Input);

    // 5. Clipped ReLU on L_1 output (input to L_2)
    ValueVector l2Input =
        ValueVector(l1Output.values.map((v) => clippedReLU(v)).toList());

    // 6. Linear Layer 2 (L_2)
    ValueVector l2Output = hiddenLayer2.forward(l2Input);

    // 7. Clipped ReLU on L_2 output (input to final output layer)
    ValueVector finalInputToOutput =
        ValueVector(l2Output.values.map((v) => clippedReLU(v)).toList());

    // 8. Output Layer (L_3)
    Value finalEvaluationCP = outputLayer.forward(finalInputToOutput).values.first;

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
      20: {'type': BoardState.KING, 'color': BoardState.WHITE}, // White King on square 20
      45: {'type': BoardState.KING, 'color': BoardState.BLACK}, // Black King on square 45
    },
    whiteKingSquare: 20,
    blackKingSquare: 45,
    sideToMove: BoardState.WHITE,
  );

  // The accumulator for the current position
  final NnueAccumulator currentPositionAccumulator =
      NnueAccumulator(numOutputsPerPerspective);

  // Perform an initial forward pass
  final nnueEvaluationCP =
      nnueModel.forward(initialBoard, currentPositionAccumulator);
  print("Initial NNUE Evaluation (CP-space): ${nnueEvaluationCP.data.toStringAsFixed(4)}");

  // Convert to WDL space for loss calculation
  final nnueEvaluationWDL = nnueModel.convertToWDL(nnueEvaluationCP);
  print("Initial NNUE Evaluation (WDL-space): ${nnueEvaluationWDL.data.toStringAsFixed(4)}");

  // --- Dummy Training Step ---
  print("\n--- Dummy Training Step ---");

  // Example: Target from dataset (e.g., true evaluation in CP) and game result
  final targetEvaluationCP = Value(200.0); // Example target CP (white is up material)
  final gameResultWDL = Value(1.0); // Example: Game resulted in a win for White

  // Convert target CP to WDL space
  final targetEvaluationWDL = nnueModel.convertToWDL(targetEvaluationCP);

  // Interpolation parameter (lambda_)
  final lambda_ = Value(0.7); // 70% evaluation, 30% game result

  // Calculate the combined target for loss
  final combinedTargetWDL =
      lambda_ * targetEvaluationWDL + (Value(1.0) - lambda_) * gameResultWDL;
  print("Combined Target (WDL-space): ${combinedTargetWDL.data.toStringAsFixed(4)}");

  // MSE Loss (exponent 2.0 for simplicity, text suggests >2 like 2.6)
  Value loss = (nnueEvaluationWDL - combinedTargetWDL).pow(2.0);

  print("Initial Loss: ${loss.data.toStringAsFixed(4)}");

  // Zero gradients, perform backward pass
  nnueModel.zeroGrad();
  loss.backward();

  // Optimizer step
  final optimizer = SGD(nnueModel.parameters(), 0.0001); // Smaller learning rate for stability
  optimizer.step();

  // --- Simulate a Move and Re-evaluate ---
  print("\n--- After 1 training step and a dummy move ---");
  final nextBoardState = initialBoard.applyMove('Ke2'); // Example king move to force refresh
  final NnueAccumulator nextPositionAccumulator =
      NnueAccumulator(numOutputsPerPerspective); // New accumulator for new position

  final nnueEvaluationCPAfterUpdate =
      nnueModel.forward(nextBoardState, nextPositionAccumulator);
  print(
      "NNUE Evaluation (CP-space) After Update: ${nnueEvaluationCPAfterUpdate.data.toStringAsFixed(4)}");

  final nnueEvaluationWDLAfterUpdate =
      nnueModel.convertToWDL(nnueEvaluationCPAfterUpdate);
  print(
      "NNUE Evaluation (WDL-space) After Update: ${nnueEvaluationWDLAfterUpdate.data.toStringAsFixed(4)}");

  Value lossAfterUpdate = (nnueEvaluationWDLAfterUpdate - combinedTargetWDL).pow(2.0);
  print("Loss After Update: ${lossAfterUpdate.data.toStringAsFixed(4)}");
  print("\nThis demonstrates the conceptual flow of NNUE inference with incremental updates");
  print("and a basic training step using WDL conversion and MSE loss.");
  print("Note: Feature factorization itself is a training-time concept (data generation/forward pass modification),");
  print("and the deployed model already has coalesced weights.");
}







Stockfish quantization scheme

Feature Transformer

Let's start with the feature transformer. Recall that its purpose is to accumulate between 0 to 30 (for HalfKP) rows of weights. We want to have int8 values as inputs to the later layers, with the activation range (ClippedReLU) being 0..127, but that means that using int8 integers for the accumulator doesn't provide enough space as the values would go beyond the range of int8 before applying the ClippedReLU... so we use int16 for the accumulator and then convert to int8 when doing the ClippedReLU.

Linear layer

We wanted int8 inputs and we can get them without losing too much precision. The nature of matrix-purposed SIMD instructions is that, thankfully, the accumulation happens in int32. So we don't experience the same issue as in the feature transformer where we're manually adding rows, and we can utilize the int8 multiplication with int32 accumulation to the fullest extent, and only later go back to int8 in the ClippedReLU layer. We will add the biases after the accumulation has happened, so they should be stored in int32.

ClippedReLU

Nothing special going on in here. Since the inputs are not being scaled, this is simply the same operation but in a different domain. Instead of clamping to 0..1 we clamp to 0..127. The input type is usually different than the output type as inputs will be either int32 or int16, and the output we want is int8. The values won't change but the conversion needs to be applied.

The math of quantization and how to make it fit

To quantize the network we need to multiply the weights and biases by some constant to translate them to a different range of values. This poses a problem when confronted with multiplication during network inference - (a*x) * (a*w) = a*a*x*w, and we have to sometimes scale back the outputs too. But each layer is still independent so let's go through them one by one again.

Feature Transformer

Remember we want our activation range to change from 0..1 to 0..127. Since the feature transformer is a purely additive process, it's enough that we multiply the weights and biases by 127. Both weights and biases are stored as int16. We could divide the output by some factor a to get more precision, in which case the weights and biases would have to be multiplied by a*127 instead, but in practice, it increases the accuracy only by a little bit.

Linear layer

To arrive at int8 weights we have to apply some scaling factor. This scaling factor ultimately depends on how much precision needs to be preserved, but cannot be too large because then the weights will be limited in magnitude. For example, if we took the scaling factor to be 64 (used in Stockfish), then the maximum weight in the floating point space is 127/64=1.984375. This is enough to have good nets, but care needs to be taken to clamp the weights during training so that they don't go outside that range. The scaling factor of 64 can also be understood as the smallest weight step that can be represented being 1/64=0.015625.

A linear layer is just matrix multiplication, so we're multiplying inputs and weights, but now both are scaled relative to the float version. Let's denote the input scaling factor (activation range scaling) as s_A, and the weight scaling factor by s_W. x is the unquantized input, w is the unquantized weight, 'b' is the unquantized bias, and y is the unquantized output. So we have:

x * w + b = y((s_A * x) * (s_W * w)) + (b * s_A * s_W) = (y * s_A) * s_W(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) / s_W = (y * s_A)

From that we learn that we need to scale the bias by (s_A * s_W), weights by s_W, and divide output by s_W to get the desired (y * s_A), which is correctly scaled to the activation range.

Now, this applies only when the next layer is the ClippedReLU layer. For the last layer, the output range is very different and the quantization will also be different. In Stockfish we want the last layer to output values in range -10000..10000 while still keeping int8 weights. This can be achieved without any additional scaling factors, but it's easiest to do and understand with an additional scaling factor.

We'll introduce a new scaling factor, s_O. This scaling factor, unlike others, needs to be applied to the output both during training (for loss calculation against the actual evaluation) and inference. The purpose of it is to scale the float output of the network to match the range of the integer evaluation used by Stockfish. Basically, it means that 1 in the float space is equal to s_O internal evaluation units. It has an additional advantage that it allows us to have the layer weights be similar in magnitude to the previous layers.

So the math is now:

x * w + b = y(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O = ((y * s_A) * s_W) * s_O(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O / s_A / s_W = (y * s_O)(((s_A * x) * (s_W / s_A * w)) + (b * s_A * s_W / s_A)) * s_O / s_W = (y * s_O)(((s_A * x) * (s_W * s_O / s_A * w)) + (b * s_W * s_O)) / s_W = (y * s_O)

From that we learn that we need to scale the bias by s_W * s_O, weights by s_W * s_O / s_A, and divide the output by s_W to get the desired (y * s_O).

Implementation

For the unoptimized implementation, not much changes. One just has to remember to change the data types to integers with desired size, scale weights on input, and divide the output from linear layers by s_W. s_W is usually chosen to be a power of two, so that this operation is a simple bitwise right shift, as there are no SIMD division instructions for integers and even if there were it would be slow.

Optimized implementation

For simplicity, we will focus on optimization only for the AVX2 extension of the x86-64 instruction set.

Feature Transformer

The benefit of SIMD for the feature transformer is two-fold:

multiple additions per instruction can be performed

large total register size means we don't need to write to memory as often

Our accumulation structure doesn't change much, we just change float to int16:

// We now also make sure that the accumulator structure is aligned to the cache line.// This is not strictly required by AVX2 instructions but may improve performance.struct alignas(64) NnueAccumulator { // Two vectors of size N. v[0] for white's, and v[1] for black's perspectives. int16_t v[2][N]; // This will be utilised in later code snippets to make the access less verbose int16_t* operator[](Color perspective) { return v[perspective]; }};

Now let's look at the refresh function. For simplicity, we will assume that there are enough registers so that spills don't happen, but in reality (M > 256) it is required to do multiple passes over the active features, each time considering a part of the accumulator only. A single AVX2 register can fit 16 int16 values and there are 16 AVX2 registers (32 since AVX-512).

void refresh_accumulator( const LinearLayer& layer, // this will always be L_0 NnueAccumulator& new_acc, // storage for the result const std::vector<int>& active_features, // the indices of features that are active for this position Color perspective // the perspective to refresh) { // The compiler should use one register per value, and hopefully // won't spill anything. Always check the assembly generated to be sure! constexpr int register_width = 256 / 16; static_assert(M % register_width == 0, "We're processing 16 elements at a time"); constexpr int num_chunks = M / register_width; __m256i regs[num_chunks]; // Load bias to registers and operate on registers only. for (int i = 0; i < num_chunks; ++i) { regs[i] = _mm256_load_si256(&layer.bias[i * register_width]); } for (int a : active_features) { for (int i = 0; i < num_chunks; ++i) { // Now we do 1 memory operation instead of 2 per loop iteration. regs[i] = _mm256_add_epi16(regs[i], _mm256_load_si256(&layer.weight[a][i * register_width])); } } // Only after all the accumulation is done do the write. for (int i = 0; i < num_chunks; ++i) { _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]); }}

similarly for the update:

void update_accumulator( const LinearLayer& layer, // this will always be L_0 NnueAccumulator& new_acc, // it's nice to have already provided storage for // the new accumulator. Relevant parts will be overwritten const NNueAccumulator& prev_acc, // the previous accumulator, the one we're reusing const std::vector<int>& removed_features, // the indices of features that were removed const std::vector<int>& added_features, // the indices of features that were added Color perspective // the perspective to update, remember we have two, // they have separate feature lists, and it even may happen // that one is updated while the other needs a full refresh) { // The compiler should use one register per value, and hopefully // won't spill anything. Always check the assembly generated to be sure! constexpr int register_width = 256 / 16; static_assert(M % register_width == 0, "We're processing 16 elements at a time"); constexpr int num_chunks = M / register_width; __m256i regs[num_chunks]; // Load the previous values to registers and operate on registers only. for (int i = 0; i < num_chunks; ++i) { regs[i] = _mm256_load_si256(&prev_acc[perspective][i * register_width]); } // Then we subtract the weights of the removed features for (int r : removed_features) { for (int i = 0; i < num_chunks; ++i) { regs[i] = _mm256_sub_epi16(regs[i], _mm256_load_si256(&layer.weight[r][i * register_width])); } } // Similar for the added features, but add instead of subtracting for (int a : added_features) { for (int i = 0; i < num_chunks; ++i) { regs[i] = _mm256_add_epi16(regs[i], _mm256_load_si256(&layer.weight[a][i * register_width])); } } // Only after all the accumulation is done do the write. for (int i = 0; i < num_chunks; ++i) { _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]); }}

Linear layer

Matrix multiplication is hard to optimize in general, and there are many approaches depending on the size of the matrices. Since we expect the layers to be small, we will not delve into any fancy blocked algorithms. And just rely on manual unrolling and trying to process multiple values at a time. This is not optimal, but it's simple and very close. We will only describe the case where the number of outputs is divisible by 4. The output layer has 1 output but it's also very small and doesn't require anything clever. We will also require the input size to be a multiple of 32, otherwise adding 0 padding is required.

int32_t* linear( const LinearLayer& layer, // the layer to use. We have two: L_1, L_2 int32_t* output, // the already allocated storage for the result const int8_t* input // the input, which is the output of the previous ClippedReLU layer) { constexpr int register_width = 256 / 8; assert(layer.num_inputs % register_width == 0, "We're processing 32 elements at a time"); assert(layer.num_outputs % 4 == 0, "We unroll by 4"); const int num_in_chunks = layer.num_inputs / register_width; const int num_out_chunks = layer.num_outputs / 4; for (int i = 0; i < num_out_chunks; ++i) { // Prepare weight offsets. One offset for one row of weights. // This is a simple index into a 2D array. const int offset0 = (i * 4 + 0) * layer.num_inputs; const int offset1 = (i * 4 + 1) * layer.num_inputs; const int offset2 = (i * 4 + 2) * layer.num_inputs; const int offset3 = (i * 4 + 3) * layer.num_inputs; // Accumulation starts from 0, we add the bias only at the end. __m256i sum0 = _mm256_setzero_si256(); __m256i sum1 = _mm256_setzero_si256(); __m256i sum2 = _mm256_setzero_si256(); __m256i sum3 = _mm256_setzero_si256(); // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time! for (int j = 0; j < num_in_chunks; ++j) { // We unroll by 4 so that we can reuse this value, reducing the number of // memory operations required. const __m256i in = _mm256_load_si256(&input[j * register_width]); // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32. // For definition see below. m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256(&layer.weights[offset0 + j * register_width])); m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256(&layer.weights[offset1 + j * register_width])); m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256(&layer.weights[offset2 + j * register_width])); m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256(&layer.weights[offset3 + j * register_width])); } const __m128i bias = _mm_load_si128(&layer.bias[i * 4]); // This function adds horizontally 8 values from each sum together, producing 4 int32 values. // For the definition see below. __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, bias); // Here we account for the weights scaling. outval = _mm_srai_epi32(outval, log2_weight_scale); _mm_store_si128(&output[i * 4], outval); } return output + layer.num_outputs;}

m256_add_dpbusd_epi32





The output needs to be horizontally accumulated further, but it's faster to do it with 4 sums (sum0, sum1, sum2, sum3) later.

This function can benefit from VNNI extension, here controlled by USE_VNNI.

void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {#if defined (USE_VNNI) // This does exactly the same thing as explained below but in one instruction. acc = _mm256_dpbusd_epi32(acc, a, b);#else // Multiply a * b and accumulate neighbouring outputs into int16 values __m256i product0 = _mm256_maddubs_epi16(a, b); // Multiply product0 by 1 (idempotent) and accumulate neighbouring outputs into int32 values __m256i one = _mm256_set1_epi16(1); product0 = _mm256_madd_epi16(product0, one); // Add to the main int32 accumulator. acc = _mm256_add_epi32(acc, product0);#endif};

m256_haddx4



This function takes 4 __m256i registers containing 8 int32 values each, accumulates them horizontally, and produces one __m128i register containing 4 int32 values, each corresponding to one input sum. In the matrix multiplication above we keep one sum per weight row/input, so in the end, we fill the output 4 values at a time.



__m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) { sum0 = _mm256_hadd_epi32(sum0, sum1); sum2 = _mm256_hadd_epi32(sum2, sum3); sum0 = _mm256_hadd_epi32(sum0, sum2); __m128i sum128lo = _mm256_castsi256_si128(sum0); __m128i sum128hi = _mm256_extracti128_si256(sum0, 1); return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);};

Linear layer with sparse input

In the previous part we described the generic dense matrix multiplication, but let's try to delve a little bit deeper. The case we will be considering here is similar to how our feature transformer operates, but here we always need to perform the full operation instead and the matrices are smaller. But why are we even considering this? Well, it turns out that the feature transformer output, after being passed through ClippedReLU, can have quite significant sparsity. Here is some data presenting the density of the inputs to the first dense fully connected layer, for networks with different feature transformer size:



(boxes correspond to the [25%, 75%] interval, whiskers correspond to the [1%, 99%] interval)

That's already <=15% density for the common sizes, and it's consistent between different networks! However we cannot make it as much faster, there is some cost related to the changed access patterns and more required preprocessing, so whether this approach works for your particular case needs to be thoroughly tested.

Let's see the code that can take advantage of it.

int lsb(std::uint32_t v) { // returns the least significant set bit in v // implementation detail // can be implemented for example using compiler intrinsics // https://www.chessprogramming.org/BitScan#Leading_Zero_Count}// This implementation requires changing the layout and expanding the weights to int16.// We will transpose the weights as now we'll be going through the columns instead of rows.void load_weights( const LinearLayer& layer, const int8_t* data) { static_assert(is_same_v<LinearLayer::WeightType, int16_t>, "This approach requires weights to be 16-bit. Otherwise, it's hard to widen the multiplication output to 32 bits."); for (int i = 0; i < layer.num_outputs; ++i) { for (int j = 0; j < layer.num_inputs; ++j) { layer.weights[j*layer.num_outputs + i] = data[i*layer.num_inputs + j]; } } // For AVX2 we must also swap some lanes in the weights. This is // because AVX2 registers functions as two 128-bit ones, and // therefore some data is interleaved in the inference process. // This makes it so that they end up where we want. // Will be more apparent in the visualization. // This effectively swaps out the middle 2 64-bit chunks in each 256-bit chunk. for (int i = 0; i < layer.num_outputs; ++i) { for (int j = 0; j < layer.num_inputs; ++j) { const int simd_lane = j % 16; const int simd_lane_64 = simd_lane / 4; if (simd_lane_64 == 1) { swap( layer.weights[i*layer.num_outputs + j + 0], layer.weights[i*layer.num_outputs + j + 4] ); } } }}int32_t* linear_sparse_input( const LinearLayer& layer, int32_t* output, const int8_t* input) { static_assert(is_same_v<LinearLayer::WeightType, int16_t>, "This approach requires weights to be 16-bit. Otherwise, it's hard to widen the multiplication output to 32 bits."); constexpr int register_width = 256 / 8; constexpr int input_register_width = register_width; // uint8_t constexpr int output_register_width = register_width / 4; // int32_t constexpr int output_chunk_size = output_register_width * 2; // we will be processing 2 registers at a time assert(layer.num_outputs % output_chunk_size == 0, "We're processing 16 output elements at a time"); assert(layer.num_inputs % input_register_width == 0); // We need to find out the indices of the input values that are non-zero uint16_t nnz_input_indices[layer.num_inputs]; int num_nnz_input_indices = 0; for (int i = 0; i < layer.num_inputs; i += input_register_width) { const __m256i input_chunk = _mm256_load_si256(input + i); // Find out where the values are greater than 0 and set the corresponding bits in nnz uint32_t nnz = _mm256_movemask_epi8( _mm256_cmpgt_epi8(input_chunk, _mm256_setzero_si256()) ); // Extract the indices of the set bits in nnz while (nnz) { const int lsb_index = lsb(nnz); nnz &= nnz - 1; // reset the least significant set bit in nnz nnz_input_indices[num_nnz_input_indices++] = i + lsb_index; } } // First we just copy the biases. Compilers are good at vectorizing this. // Could also use memcpy for (int i = 0; i < layer.num_outputs; ++i) { output[i] = layer.biases[i]; } const int num_chunks = layer.num_outputs / output_chunk_size; int i = 0; for (; i + 1 < num_nnz_input_indices; i += 2) { // We will try to process 2 at a time as much as possible, // as we can utilize the available intrinsics better. // Will become more apparent on the visualization. const int input_id0 = nnz_input_indices[i+0]; const int input_id1 = nnz_input_indices[i+1]; const __m256i factor = _mm256_set1_epi32( input[input_id0] | (input[input_id1] << 16) ); for (int j = 0; j < num_chunks; ++j) { const int output_offset0 = (j*2 + 0)*output_register_width; const int output_offset1 = (j*2 + 1)*output_register_width; // Weights are packed 2 times as densely as the output. const int weight_offset = (j*1 + 0)*output_register_width; // Each chunk requires a load+store. // However, if the output is small enough it can be unrolled and // all outputs might fit into the registers. // Though the compiler probably is not allowed to do it by itself. __m256i sum0 = _mm256_load_si256(output + output_offset0); __m256i sum1 = _mm256_load_si256(output + output_offset1); // Remember, weights are 16 bit here, so one __m256i can hold 16 of them. const __m256i col0 = _mm256_load_si256( layer.weights + input_id0 * layer.num_outputs + weight_offset ); const __m256i col1 = _mm256_load_si256( layer.weights + input_id1 * layer.num_outputs + weight_offset ); // See next below for visualization m256_process_chunk(sum0, sum1, col0, col1, factor); _mm256_store_si256(output + output_offset0, sum0); _mm256_store_si256(output + output_offset1, sum1); } } // Process the remaining single input for (; i < num_nnz_input_indices; ++i) { const int input_id = nnz_input_indices[i]; const __m256i factor = _mm256_set1_epi32(input[input_id]); for (int j = 0; j < num_chunks; ++j) { const int output_offset0 = (j*2 + 0)*output_register_width; const int output_offset1 = (j*2 + 1)*output_register_width; const int weight_offset = (j*1 + 0)*output_register_width; __m256i sum0 = _mm256_load_si256(output + output_offset0); __m256i sum1 = _mm256_load_si256(output + output_offset1); const __m256i col0 = _mm256_load_si256( layer.weights + input_id * layer.num_outputs + weight_offset ); m256_process_chunk(sum0, sum1, col0, _mm256_setzero_si256(), factor); _mm256_store_si256(output + output_offset0, sum0); _mm256_store_si256(output + output_offset1, sum1); } } for (int j = 0; j < layer.num_outputs; j += output_register_width) { _mm256_store_si256(output + j, _mm256_srai_epi32(_mm256_load_si256(output + j), log2_weight_scale)); } return output + layer.num_outputs;}

m256_process_chunk



This function takes int16 weights, a factor being a composition of 2 int8 inputs broadcasted as int32, and produces int32 outputs.



inline void m256_process_chunk(__m256i& sum0, __m256i& sum1, __m256i col0, __m256i col1, __m256i factor) { // We interleave the two columns, because madd adds adjacent values. // This way we effectively add the results from both columns. sum0 = _mm256_add_epi32( sum0, _mm256_madd_epi16(factor, _mm256_unpacklo_epi16(col0, col1)) ); sum1 = _mm256_add_epi32( sum1, _mm256_madd_epi16(factor, _mm256_unpackhi_epi16(col0, col1)) );}

Linear layer with sparse input, alternative approach

In the first approach, we used 16-bit weights, but it's possible to use 8-bit weights with slightly more unpacking fun. We'll also see an alternative way of computing the indices on non-zero inputs by using a lookup table. For some more approaches and measurements to the latter see here.

// This implementation requires changing the layout and expanding the weights to int16.// We will transpose the weights as now we'll be going through the columns instead of rows.void load_weights( const LinearLayer& layer, const int8_t* data) { static_assert(is_same_v<LinearLayer::WeightType, int8_t>, "This approach requires weights to be 8-bit."); for (int i = 0; i < layer.num_outputs; ++i) { for (int j = 0; j < layer.num_inputs; ++j) { layer.weights[j*layer.num_outputs + i] = data[i*layer.num_inputs + j]; } } // No need for clever tricks with shuffling the weights now. // However, we will require one more zero-weight column. We assume enough space is allocated. for (int i = 0; i < layer.num_outputs; ++i) { layer.weights[layer.num_inputs*layer.num_outputs + i] = 0; }}// A constexpr version of least significant bit computation.static constexpr int lsb_constexpr(std::uint32_t v){ int c = 0; if (!v) return 32; while (!(v & 1)) { v >>= 1; ++c; } return c;}// A lookup table of indices of non-zero bits in the input.// Each entry of std::array<std::uint16_t, 8> can be interpreted as __m128i.alignas(64) static constexpr std::array<std::array<std::uint16_t, 8>, 256> LookupTableIndices = [](){ std::array<std::array<std::uint16_t, 8>, 256> v{}; for (int i = 0; i < 256; ++i) { int j = i; int k = 0; while(j) { const IndexType lsbIndex = lsb_constexpr(std::uint32_t(j)); j &= j - 1; v[i][k] = lsbIndex; ++k; } } return v;}();// A lookup table for popcount of a byte.// Using the dedicated popcnt instruction might or might not work better.static constexpr std::array<std::uint8_t, 256> LookupTableCounts = [](){ std::array<std::uint8_t, 256> v{}; for (int i = 0; i < 256; ++i) { int j = i; int k = 0; while(j) { j &= j - 1; ++k; } v[i] = k; } return v;}();int32_t* linear_sparse_input( const LinearLayer& layer, int32_t* output, const int8_t* input) { // We will take a tiled approach with accumulators in registers. // Similar to how the feature transformer is best implemented. constexpr int input_register_width = 256 / 8; constexpr int chunk_size = 256 / 32; constexpr int num_chunks_per_tile = 8; constexpr int tile_size = chunk_size * num_chunks_per_tile; assert(layer.num_outputs % tile_size == 0, "We're processing 64 output elements at a time. Though it's easy to change it."); assert(num_chunks_per_tile % 4 == 0, "We're processing 4 chunks at a time."); constexpr int num_tiles = layer.num_outputs / tile_size; // We need to find out the indices of the input values that are non-zero // We'll use a lookup table approach. Overallocate 16 elements // so that stores are always valid (we will be using larger stores) uint16_t nnz_input_indices[layer.num_inputs + 16]; int num_nnz_input_indices = 0; { // These will be used for offsetting the looked-up indices. // A variation with int16 lookup is also possible (see the link above) // and is faster in isolation, but requires more memory and may trash the cache. __m128i base = _mm_set1_epi16(0); __m128i increment = _mm_set1_epi16(8); for (int i = 0; i < layer.num_inputs; i += input_register_width) { const __m256i input_chunk = _mm256_load_si256(input + i); unsigned nnz = _mm256_movemask_epi8(_mm256_cmpgt_epi8(input_chunk, _mm256_setzero_si256())); unsigned b0 = (nnz) & 0xFF; unsigned b1 = (nnz >> 8) & 0xFF; unsigned b2 = (nnz >> 16) & 0xFF; unsigned b3 = (nnz >> 24) & 0xFF; unsigned c0 = LookupTableCounts[b0]; unsigned c1 = LookupTableCounts[b1]; unsigned c2 = LookupTableCounts[b2]; unsigned c3 = LookupTableCounts[b3]; // These stores can reach above layer.num_inputs in extreme cases. That's why we preallocate. // Only the first c0 values matter. _mm_storeu_si128( reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices), _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b0])), base) ); num_nnz_input_indices += c0; base = _mm_add_epi32(base, increment); _mm_storeu_si128( reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices), _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b1])), base) ); num_nnz_input_indices += c1; base = _mm_add_epi32(base, increment); _mm_storeu_si128( reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices), _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b2])), base) ); num_nnz_input_indices += c2; base = _mm_add_epi32(base, increment); _mm_storeu_si128( reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices), _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b3])), base) ); num_nnz_input_indices += c3; base = _mm_add_epi32(base, increment); } } // We will be processing 4 inputs at a time, and to avoid having two similar loops // we pad the input indices to a multiple of 4. For the added ones we use a dummy input // with all weights set to 0. while (num_nnz_input_indices % 4 != 0) nnz_input_indices[num_nnz_input_indices++] = layer.num_inputs; // Hopefully will fit in the register file. __m256i acc[num_chunks_per_tile]; for (int i = 0; i < num_tiles; ++i) { const __m256i* biases_tile = reinterpret_cast<const __m256i*>(&layer.biases[i * tile_size]); __m256i* output_tile = reinterpret_cast< __m256i*>(& output[i * tile_size]); for (int k = 0; k < num_chunks_per_tile; ++k) acc[k] = _mm256_setzero_si256(); for (int j = 0; j < num_nnz_input_indices; j += 4) { const __m256i mul0 = _mm256_set1_epi16(input[nnz_input_indices[j+0]] | (input[nnz_input_indices[j+1]] << 8)); const __m256i mul2 = _mm256_set1_epi16(input[nnz_input_indices[j+2]] | (input[nnz_input_indices[j+3]] << 8)); const __m256i* col0 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+0] * layer.num_outputs + i * tile_size]); const __m256i* col1 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+1] * layer.num_outputs + i * tile_size]); const __m256i* col2 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+2] * layer.num_outputs + i * tile_size]); const __m256i* col3 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+3] * layer.num_outputs + i * tile_size]); for (int k = 0; k < num_chunks_per_tile / 4; ++k) { // Due to AVX2 interpreting the 256-bit registers as 2 128-bit ones the unpacking // shuffles the lanes. We will have to account for that when getting the final result. m256_process_chunk_alternative( acc[k*4 + 0], acc[k*4 + 1], acc[k*4 + 2], acc[k*4 + 3], col0[k], col1[k], col2[k], col3[k], mul0, mul2 ); } } for (int k = 0; k < num_chunks_per_tile / 4; ++k) { // We have to unshuffle the lanes. See the visualization to get a better picture. const __m128i acc00 = _mm256_extracti128_si256(acc[k*4 + 0], 0); const __m128i acc01 = _mm256_extracti128_si256(acc[k*4 + 0], 1); const __m128i acc10 = _mm256_extracti128_si256(acc[k*4 + 1], 0); const __m128i acc11 = _mm256_extracti128_si256(acc[k*4 + 1], 1); const __m128i acc20 = _mm256_extracti128_si256(acc[k*4 + 2], 0); const __m1




Show thinking
The provided text describes the Stockfish quantization scheme for its Neural Network Updatable Evaluation (NNUE) model, focusing on the data types, scaling factors, and optimized implementation details using AVX2 intrinsics.

Stockfish Quantization Scheme
The core idea of quantization is to represent floating-point numbers with integers to improve performance and reduce memory footprint, especially on CPUs.

Data Types and Flow:

Feature Transformer (L0):

Input: Active features.

Accumulator: int16. This is chosen because int8 would overflow before applying the activation, as it needs to sum multiple feature weights (up to 30 for HalfKP).

Output: The int16 accumulator values.

Linear Layers (L1, L2, etc.):

Input: int8 (from the previous ClippedReLU).

Weights: int8.

Accumulation (internal): int32. This handles the int8 * int8 multiplication and subsequent additions without overflow, leveraging SIMD instructions that accumulate into larger registers.

Biases: int32. Added after the main accumulation.

Output: The int32 accumulator values.

Clipped ReLU:

Input: int16 (from Feature Transformer) or int32 (from Linear Layer).

Operation: Clamps values to 0..127 (instead of 0..1 in float space).

Output: int8. This converts the larger integer types back to int8 for the next layer's input.

The Math of Quantization and Scaling Factors
To translate values between the original floating-point space and the quantized integer space, various scaling factors are applied:

s_A (Input/Activation Scaling Factor): This scales the activation range. If the original float activation is 0..1, the quantized int8 activation becomes 0..127. Thus, s_A = 127.

s_W (Weight Scaling Factor): Applied to the weights of linear layers to convert them to int8. For example, Stockfish uses s_W = 64. This means a floating-point weight w becomes w * s_W (rounded to int8). The maximum representable float weight for int8 (max 127) with s_W=64 is 127/64 ≈ 1.98.

s_O (Output Scaling Factor for the Last Layer): This special scaling factor is applied to the final output of the network (e.g., the L3 output layer). Its purpose is to map the internal float evaluation of the network to the expected integer centipawn scale (e.g., -10000..10000 in Stockfish). If 1 in float space corresponds to s_O internal evaluation units, then the network's final output y (in float space) becomes y * s_O after quantization.

Scaling Rules for Layers:

Feature Transformer: Weights and biases are multiplied by 127 to get int16 values. The output (accumulator) is already in the 0..127 range after being conceptualized as such during the scaling.

Linear Layer (intermediate, outputting to ClippedReLU):

Input: x_quant = x * s_A

Weights: w_quant = w * s_W

Biases: b_quant = b * s_A * s_W

Output (before scaling): (x_quant * w_quant) + b_quant (accumulated in int32)

Final Output (after scaling): ((x_quant * w_quant) + b_quant) / s_W. This results in y * s_A, which is correctly scaled for the next ClippedReLU layer. Dividing by s_W is typically a bitwise right shift if s_W is a power of two (e.g., 64).

Linear Layer (final, outputting final evaluation):

Input: x_quant = x * s_A

Weights: w_quant = w * s_W * s_O / s_A

Biases: b_quant = b * s_W * s_O

Output (before scaling): (x_quant * w_quant) + b_quant (accumulated in int32)

Final Output (after scaling): ((x_quant * w_quant) + b_quant) / s_W. This results in y * s_O, which is the final evaluation scaled to the desired output range.

Optimized Implementation (AVX2 Intrinsics)
The provided C++ code snippets illustrate highly optimized low-level implementations using AVX2 (Advanced Vector Extensions 2), an instruction set for x86-64 CPUs. These optimizations are designed to leverage SIMD (Single Instruction, Multiple Data) capabilities, processing multiple data elements simultaneously.

Feature Transformer:

Uses int16 arrays for accumulators and weights, aligned to cache lines for performance.

Leverages _mm256_load_si256 to load 16 int16 values into a 256-bit AVX2 register (__m256i).

Uses _mm256_add_epi16 for parallel int16 additions, performing 16 additions per instruction.

Minimizes memory access by loading biases into registers first, accumulating, and only writing back to memory once.

Linear Layer (Dense Input):

Uses int8 inputs and weights, with internal int32 accumulation.

Unrolls loops and processes multiple output neurons (e.g., 4 at a time) and multiple input chunks (e.g., 32 at a time) to maximize SIMD utilization.

_mm256_maddubs_epi16 (part of m256_add_dpbusd_epi32 or directly _mm256_dpbusd_epi32 with VNNI) performs int8 multiplication and int16 accumulation, and then _mm256_madd_epi16 aggregates int16 to int32.

m256_haddx4 performs horizontal sums across 256-bit registers to combine partial int32 sums efficiently.

_mm_srai_epi32 performs arithmetic right shift for the division by s_W.

Linear Layer (Sparse Input):

This is a more advanced optimization for cases where the input (output of ClippedReLU) is sparse (many zeros), as shown in the density chart.

It involves transposing weights, identifying non-zero input indices using bit-masking (_mm256_movemask_epi8) and lookup tables (LookupTableIndices, LookupTableCounts), and then performing multiplications only for the active inputs.

The m256_process_chunk_alternative function uses _mm256_unpacklo_epi16 and _mm256_unpackhi_epi16 to interleave and combine weight columns, allowing efficient processing of multiple inputs (e.g., 2 or 4 at a time) with _mm256_madd_epi16.

Dart Implementation Context
The Dart code provided in the previous response is a conceptual implementation of the NNUE model's forward and backward passes using double (floating-point) values and a custom autograd engine. It does not implement the low-level integer quantization or SIMD optimizations described in the Stockfish scheme.

Quantization: The Dart NNUEModel currently uses double throughout. To implement quantization, the Value class and all arithmetic operations would need to be re-engineered to work with fixed-point integers (int8, int16, int32) and apply the explicit scaling factors (s_A, s_W, s_O) at each layer's boundaries. This involves careful integer arithmetic and bit shifting.

SIMD Optimization (AVX2): Dart, as a high-level language, does not provide direct access to CPU intrinsics like AVX2. The C++ code snippets showcasing _mm256_load_si256, _mm256_add_epi16, _mm256_maddubs_epi16, etc., are specific to x86-64 assembly extensions and cannot be directly translated or implemented in standard Dart. Achieving similar performance in Dart would typically require:

FFI (Foreign Function Interface): Calling highly optimized C/C++ code (like the Stockfish implementation) from Dart.

Vectorization Libraries: Relying on underlying Dart VM optimizations or specialized libraries that internally use SIMD instructions, but these are generally higher-level abstractions.

The provided Dart code aims to illustrate the NNUE architecture and the conceptual flow of forward pass, loss calculation, and a basic training step using floating-point values and automatic differentiation, which is the standard approach for training deep learning models before applying quantization for deployment.