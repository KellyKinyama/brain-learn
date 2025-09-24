// main.dart
import 'dart:io';
import 'dart:math';
import 'package:path/path.dart' as p;
import 'ars_learner.dart';
import 'gym_mock.dart'; // Mock Gym environment
import 'policies.dart'; // Policy and filter implementations
import 'shared_noise.dart'; // Shared noise table
import 'optimizers.dart'; // Optimizer implementations

/// Parameters for the ARS algorithm.
class ARSParams {
  final String envName;
  final int nIter;
  final int nDirections;
  final int deltasUsed;
  final double stepSize;
  final double deltaStd;
  final int nWorkers;
  final int rolloutLength;
  final double shift;
  final int seed;
  final String policyType;
  final String dirPath;
  final String filter;

  ARSParams({
    required this.envName,
    required this.nIter,
    required this.nDirections,
    required this.deltasUsed,
    required this.stepSize,
    required this.deltaStd,
    required this.nWorkers,
    required this.rolloutLength,
    required this.shift,
    required this.seed,
    required this.policyType,
    required this.dirPath,
    required this.filter,
  });

  Map<String, dynamic> toJson() => {
        'env_name': envName,
        'n_iter': nIter,
        'n_directions': nDirections,
        'deltas_used': deltasUsed,
        'step_size': stepSize,
        'delta_std': deltaStd,
        'n_workers': nWorkers,
        'rollout_length': rolloutLength,
        'shift': shift,
        'seed': seed,
        'policy_type': policyType,
        'dir_path': dirPath,
        'filter': filter,
      };
}

/// Runs the ARS algorithm with the given parameters.
Future<void> runARS(ARSParams params) async {
  final dir = Directory(params.dirPath);
  if (!await dir.exists()) {
    await dir.create(recursive: true);
  }
  final logdir = params.dirPath; // Using dirPath as logdir for simplicity

  // Mock environment for getting observation and action dimensions
  final GymEnv env;
  if (params.envName == 'HalfCheetah-v1') {
    env = MockHalfCheetahEnv();
  } else if (params.envName == 'Chess-v0') {
    env = MockChessEnv(); // Instantiate the mock chess environment
  }
  else {
    throw UnimplementedError('Environment ${params.envName} not mocked.');
  }

  final int obDim = env.observationSpace.shape[0];
  final int acDim = env.actionSpace.shape[0];

  // Set policy parameters
  final Map<String, dynamic> policyParams = {
    'type': params.policyType,
    'ob_filter': params.filter,
    'ob_dim': obDim,
    'ac_dim': acDim,
  };

  final ars = ARSLearner(
    envName: params.envName,
    policyParams: policyParams,
    numWorkers: params.nWorkers,
    numDeltas: params.nDirections,
    deltasUsed: params.deltasUsed,
    stepSize: params.stepSize,
    deltaStd: params.deltaStd,
    logdir: logdir,
    rolloutLength: params.rolloutLength,
    shift: params.shift,
    params: params.toJson(),
    seed: params.seed,
  );

  await ars.train(params.nIter);
  env.close(); // Close mock environment
}

void main(List<String> arguments) async {
  // Simple argument parsing (can be extended with a package like 'args')
  String envName = 'HalfCheetah-v1';
  int nIter = 1000;
  int nDirections = 8;
  int deltasUsed = 8;
  double stepSize = 0.02;
  double deltaStd = 0.03;
  int nWorkers = 18;
  int rolloutLength = 1000;
  double shift = 0.0; // For HalfCheetah-v1, shift = 0
  int seed = 237;
  String policyType = 'linear';
  String dirPath = 'data';
  String filter = 'MeanStdFilter'; // For ARS V2, use MeanStdFilter

  // Parse command line arguments
  for (var arg in arguments) {
    if (arg.startsWith('--env_name=')) {
      envName = arg.substring('--env_name='.length);
    } else if (arg.startsWith('--n_iter=')) {
      nIter = int.parse(arg.substring('--n_iter='.length));
    } else if (arg.startsWith('--n_directions=')) {
      nDirections = int.parse(arg.substring('--n_directions='.length));
    } else if (arg.startsWith('--deltas_used=')) {
      deltasUsed = int.parse(arg.substring('--deltas_used='.length));
    } else if (arg.startsWith('--step_size=')) {
      stepSize = double.parse(arg.substring('--step_size='.length));
    } else if (arg.startsWith('--delta_std=')) {
      deltaStd = double.parse(arg.substring('--delta_std='.length));
    } else if (arg.startsWith('--n_workers=')) {
      nWorkers = int.parse(arg.substring('--n_workers='.length));
    } else if (arg.startsWith('--rollout_length=')) {
      rolloutLength = int.parse(arg.substring('--rollout_length='.length));
    } else if (arg.startsWith('--shift=')) {
      shift = double.parse(arg.substring('--shift='.length));
    } else if (arg.startsWith('--seed=')) {
      seed = int.parse(arg.substring('--seed='.length));
    } else if (arg.startsWith('--policy_type=')) {
      policyType = arg.substring('--policy_type='.length);
    } else if (arg.startsWith('--dir_path=')) {
      dirPath = arg.substring('--dir_path='.length);
    } else if (arg.startsWith('--filter=')) {
      filter = arg.substring('--filter='.length);
    }
  }

  final params = ARSParams(
    envName: envName,
    nIter: nIter,
    nDirections: nDirections,
    deltasUsed: deltasUsed,
    stepSize: stepSize,
    deltaStd: deltaStd,
    nWorkers: nWorkers,
    rolloutLength: rolloutLength,
    shift: shift,
    seed: seed,
    policyType: policyType,
    dirPath: dirPath,
    filter: filter,
  );

  await runARS(params);
}

// ars_learner.dart
import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'package:vector_math/vector_math.dart';
import 'gym_mock.dart';
import 'policies.dart';
import 'shared_noise.dart';
import 'optimizers.dart';

/// Represents a message sent to a worker Isolate.
class WorkerMessage {
  final Float64List policyWeights;
  final int numRollouts;
  final double shift;
  final bool evaluate;

  WorkerMessage({
    required this.policyWeights,
    required this.numRollouts,
    required this.shift,
    required this.evaluate,
  });
}

/// Represents the result received from a worker Isolate.
class WorkerResult {
  final List<int> deltasIdx;
  final List<List<double>> rolloutRewards;
  final int steps;
  final Float64List? filterMean;
  final Float64List? filterStd;
  final int? filterCount;

  WorkerResult({
    required this.deltasIdx,
    required this.rolloutRewards,
    required this.steps,
    this.filterMean,
    this.filterStd,
    this.filterCount,
  });
}

/// Object class for parallel rollout generation, designed to run in an Isolate.
class Worker {
  final int envSeed;
  final String envName;
  final Map<String, dynamic> policyParams;
  final int noiseTableSeed;
  final int rolloutLength;
  final double deltaStd;

  late GymEnv env;
  late SharedNoiseTable deltas;
  late Policy policy;

  Worker(
    this.envSeed,
    this.envName,
    this.policyParams,
    this.noiseTableSeed,
    this.rolloutLength,
    this.deltaStd,
  );

  /// Initializes the worker's environment and policy.
  void initialize() {
    if (envName == 'HalfCheetah-v1') {
      env = MockHalfCheetahEnv();
    } else if (envName == 'Chess-v0') {
      env = MockChessEnv(); // Initialize the mock chess environment
    }
    else {
      throw UnimplementedError('Environment $envName not mocked.');
    }
    env.seed(envSeed);

    deltas = SharedNoiseTable.fromSeed(noiseTableSeed);

    if (policyParams['type'] == 'linear') {
      policy = LinearPolicy(policyParams);
    } else {
      throw UnimplementedError('Policy type ${policyParams['type']} not implemented.');
    }
  }

  /// Performs one rollout of maximum length rolloutLength.
  /// At each time-step it subtracts shift from the reward.
  List<double> rollout({double shift = 0.0, int? rolloutLength}) {
    rolloutLength ??= this.rolloutLength;

    double totalReward = 0.0;
    int steps = 0;

    Float64List ob = env.reset();
    for (int i = 0; i < rolloutLength; i++) {
      // For discrete action spaces like chess, the policy's act method
      // would need to return a discrete action (e.g., an int representing a move index).
      // The environment's step method would then take this discrete action.
      // This is a placeholder for that logic.
      Float64List action = policy.act(ob);
      var result = env.step(action); // This step expects a Float64List, adapt for discrete
      ob = result.observation;
      double reward = result.reward;
      bool done = result.done;

      steps++;
      totalReward += (reward - shift);
      if (done) {
        break;
      }
    }
    return [totalReward, steps.toDouble()];
  }

  /// Generates multiple rollouts with a policy parametrized by w_policy.
  WorkerResult doRollouts(
      Float64List wPolicy, int numRollouts, double shift, bool evaluate) {
    List<List<double>> rolloutRewards = [];
    List<int> deltasIdx = [];
    int totalSteps = 0;

    for (int i = 0; i < numRollouts; i++) {
      if (evaluate) {
        policy.updateWeights(wPolicy);
        deltasIdx.add(-1);

        // Set to false so that evaluation rollouts are not used for updating state statistics
        policy.updateFilter = false;

        // For evaluation we do not shift the rewards (shift = 0) and we use the
        // default rollout length (1000 for the MuJoCo locomotion tasks)
        List<double> rewardAndSteps =
            rollout(shift: 0.0, rolloutLength: env.spec.timestepLimit);
        rolloutRewards.add([rewardAndSteps[0]]); // Only total reward for evaluation
      } else {
        var deltaData = deltas.getDelta(wPolicy.length);
        int idx = deltaData.item1;
        Float64List delta = deltaData.item2;

        delta = (delta * deltaStd); // Apply delta_std
        // Reshape is not directly needed for 1D Float64List, but conceptually it's here
        // if wPolicy was a matrix, this would be more complex.

        deltasIdx.add(idx);

        // Set to true so that state statistics are updated
        policy.updateFilter = true;

        // compute reward and number of timesteps used for positive perturbation rollout
        policy.updateWeights(wPolicy + delta);
        List<double> posRewardAndSteps = rollout(shift: shift);
        double posReward = posRewardAndSteps[0];
        int posSteps = posRewardAndSteps[1].toInt();

        // compute reward and number of timesteps used for negative pertubation rollout
        policy.updateWeights(wPolicy - delta);
        List<double> negRewardAndSteps = rollout(shift: shift);
        double negReward = negRewardAndSteps[0];
        int negSteps = negRewardAndSteps[1].toInt();

        totalSteps += posSteps + negSteps;
        rolloutRewards.add([posReward, negReward]);
      }
    }

    return WorkerResult(
      deltasIdx: deltasIdx,
      rolloutRewards: rolloutRewards,
      steps: totalSteps,
      filterMean: policy.observationFilter.mean,
      filterStd: policy.observationFilter.std,
      filterCount: policy.observationFilter.count,
    );
  }

  /// Increments the observation filter statistics.
  void statsIncrement() {
    policy.observationFilter.statsIncrement();
  }

  /// Gets the current policy weights.
  Float64List getWeights() {
    return policy.getWeights();
  }

  /// Gets the observation filter.
  ObservationFilter getFilter() {
    return policy.observationFilter;
  }

  /// Synchronizes the worker's observation filter with another filter.
  void syncFilter(ObservationFilter otherFilter) {
    policy.observationFilter.sync(otherFilter);
  }
}

/// Entry point for the worker Isolate.
void workerEntryPoint(SendPort sendPort) {
  late Worker worker;
  ReceivePort receivePort = ReceivePort();
  sendPort.send(receivePort.sendPort); // Send back the receive port for this isolate

  receivePort.listen((message) async {
    if (message is List && message[0] == 'init') {
      // Initialize the worker
      worker = Worker(
        message[1] as int, // envSeed
        message[2] as String, // envName
        message[3] as Map<String, dynamic>, // policyParams
        message[4] as int, // noiseTableSeed
        message[5] as int, // rolloutLength
        message[6] as double, // deltaStd
      );
      worker.initialize();
      sendPort.send('initialized');
    } else if (message is WorkerMessage) {
      // Perform rollouts
      WorkerResult result = await worker.doRollouts(
        message.policyWeights,
        message.numRollouts,
        message.shift,
        message.evaluate,
      );
      sendPort.send(result);
    } else if (message == 'get_weights_plus_stats') {
      // Get policy weights and filter stats
      sendPort.send({
        'weights': worker.getWeights(),
        'filter_mean': worker.policy.observationFilter.mean,
        'filter_std': worker.policy.observationFilter.std,
        'filter_count': worker.policy.observationFilter.count,
      });
    } else if (message == 'stats_increment') {
      worker.statsIncrement();
      sendPort.send('stats_incremented');
    } else if (message is ObservationFilter) {
      // Sync filter
      worker.syncFilter(message);
      sendPort.send('filter_synced');
    } else if (message == 'get_filter') {
      // Get filter object (for master to update its own filter)
      sendPort.send(worker.getFilter());
    }
  });
}

/// Object class implementing the ARS algorithm.
class ARSLearner {
  final String envName;
  final Map<String, dynamic> policyParams;
  final int numWorkers;
  final int numDeltas;
  final int deltasUsed;
  final double deltaStd;
  final String logdir;
  final int rolloutLength;
  final double stepSize;
  final double shift;
  final Map<String, dynamic> params;
  final int seed;

  late int timesteps;
  late int actionSize;
  late int obSize;
  late SharedNoiseTable deltas;
  late Policy policy;
  late SGD optimizer;
  late List<_WorkerHandle> workers;

  ARSLearner({
    required this.envName,
    required this.policyParams,
    required this.numWorkers,
    required this.numDeltas,
    required this.deltasUsed,
    required this.deltaStd,
    required this.logdir,
    required this.rolloutLength,
    required this.stepSize,
    required this.shift,
    required this.params,
    required this.seed,
  }) {
    // Mock environment for getting dimensions
    final GymEnv env;
    if (envName == 'HalfCheetah-v1') {
      env = MockHalfCheetahEnv();
    } else if (envName == 'Chess-v0') {
      env = MockChessEnv(); // Instantiate the mock chess environment
    }
    else {
      throw UnimplementedError('Environment $envName not mocked.');
    }
    actionSize = env.actionSpace.shape[0];
    obSize = env.observationSpace.shape[0];
    env.close(); // Close mock env

    timesteps = 0;

    // Create shared table for storing noise
    print("Creating deltas table.");
    deltas = SharedNoiseTable.fromSeed(seed + 3);
    print('Created deltas table.');

    // Initialize policy
    if (policyParams['type'] == 'linear') {
      policy = LinearPolicy(policyParams);
    } else {
      throw UnimplementedError('Policy type ${policyParams['type']} not implemented.');
    }

    // Initialize optimization algorithm
    optimizer = SGD(policy.getWeights(), stepSize);

    print("Initialization of ARS complete.");
  }

  /// Initializes all worker Isolates.
  Future<void> _initializeWorkers() async {
    print('Initializing workers.');
    workers = [];
    for (int i = 0; i < numWorkers; i++) {
      final workerHandle = _WorkerHandle();
      await workerHandle.spawn(
        envSeed: seed + 7 * i,
        envName: envName,
        policyParams: policyParams,
        noiseTableSeed: seed + 7, // All workers share the same noise table instance
        rolloutLength: rolloutLength,
        deltaStd: deltaStd,
      );
      workers.add(workerHandle);
    }
    print('Workers initialized.');
  }

  /// Aggregates update step from rollouts generated in parallel.
  Future<Float64List> aggregateRollouts({int? numRollouts, bool evaluate = false}) async {
    numRollouts ??= numDeltas;

    final stopwatch = Stopwatch()..start();

    final int rolloutsPerWorker = (numRollouts / numWorkers).floor();
    final int remainingRollouts = numRollouts % numWorkers;

    final List<Future<WorkerResult>> rolloutFutures = [];

    // Distribute rollouts among workers
    for (int i = 0; i < numWorkers; i++) {
      rolloutFutures.add(workers[i].doRollouts(
        policy.getWeights(),
        rolloutsPerWorker,
        shift,
        evaluate,
      ));
    }

    // Assign remaining rollouts to the first few workers
    for (int i = 0; i < remainingRollouts; i++) {
      rolloutFutures.add(workers[i].doRollouts(
        policy.getWeights(),
        1,
        shift,
        evaluate,
      ));
    }

    final List<WorkerResult> results = await Future.wait(rolloutFutures);
    stopwatch.stop();
    print('Time to generate rollouts: ${stopwatch.elapsed.inMilliseconds / 1000.0} s');

    List<int> allDeltasIdx = [];
    List<List<double>> allRolloutRewards = [];
    List<ObservationFilter> workerFilters = [];

    for (var result in results) {
      if (!evaluate) {
        timesteps += result.steps;
      }
      allDeltasIdx.addAll(result.deltasIdx);
      allRolloutRewards.addAll(result.rolloutsRewards);
      // Reconstruct filter from worker data
      if (result.filterMean != null && result.filterStd != null && result.filterCount != null) {
        workerFilters.add(MeanStdFilter(
          obSize,
          mean: result.filterMean!,
          std: result.filterStd!,
          count: result.filterCount!,
        ));
      }
    }

    // Update master filter with worker statistics
    if (!evaluate) {
      for (var workerFilter in workerFilters) {
        policy.observationFilter.update(workerFilter);
      }
    }

    if (evaluate) {
      final rewards = allRolloutRewards.map((r) => r[0]).toList();
      print('Maximum reward of collected rollouts: ${rewards.reduce(max)}');
      return Float64List.fromList(rewards);
    }

    print('Maximum reward of collected rollouts: ${allRolloutRewards.map((r) => r.reduce(max)).reduce(max)}');

    // Select top performing directions if deltasUsed < numDeltas
    List<double> maxRewards = allRolloutRewards.map((r) => r.reduce(max)).toList();
    List<int> sortedIndices = List.generate(maxRewards.length, (i) => i)
      ..sort((a, b) => maxRewards[b].compareTo(maxRewards[a]));

    int actualDeltasUsed = min(deltasUsed, numDeltas);
    List<int> selectedIndices = sortedIndices.sublist(0, actualDeltasUsed);

    List<int> selectedDeltasIdx = selectedIndices.map((i) => allDeltasIdx[i]).toList();
    List<List<double>> selectedRolloutRewards = selectedIndices.map((i) => allRolloutRewards[i]).toList();

    // Normalize rewards by their standard deviation
    final List<double> flatRewards = selectedRolloutRewards.expand((r) => r).toList();
    final double stdRewards = _calculateStdDev(flatRewards);
    if (stdRewards > 1e-6) { // Avoid division by zero
      for (int i = 0; i < selectedRolloutRewards.length; i++) {
        selectedRolloutRewards[i][0] /= stdRewards;
        selectedRolloutRewards[i][1] /= stdRewards;
      }
    }

    stopwatch.reset();
    stopwatch.start();

    // Aggregate rollouts to form g_hat, the gradient used to compute SGD step
    Float64List gHat = Float64List(policy.getWeights().length);
    for (int i = 0; i < selectedRolloutRewards.length; i++) {
      double rewardDiff = selectedRolloutRewards[i][0] - selectedRolloutRewards[i][1];
      int deltaIdx = selectedDeltasIdx[i];
      Float64List delta = deltas.get(deltaIdx, policy.getWeights().length);
      for (int j = 0; j < gHat.length; j++) {
        gHat[j] += rewardDiff * delta[j];
      }
    }
    gHat = (gHat / selectedDeltasIdx.length); // Average over deltas used

    stopwatch.stop();
    print('Time to aggregate rollouts: ${stopwatch.elapsed.inMilliseconds / 1000.0} s');
    return gHat;
  }

  /// Performs one update step of the policy weights.
  Future<void> trainStep() async {
    Float64List gHat = await aggregateRollouts();
    print("Euclidean norm of update step: ${gHat.norm()}");
    Float64List updateStep = optimizer.computeStep(gHat);
    policy.updateWeights(policy.getWeights() - updateStep);
  }

  /// Trains the ARS algorithm for a given number of iterations.
  Future<void> train(int numIter) async {
    await _initializeWorkers(); // Initialize workers before training loop

    final overallStopwatch = Stopwatch()..start();
    for (int i = 0; i < numIter; i++) {
      final stepStopwatch = Stopwatch()..start();
      await trainStep();
      stepStopwatch.stop();
      print('Total time of one step: ${stepStopwatch.elapsed.inMilliseconds / 1000.0} s');
      print('Iter $i done');

      // Record statistics every 10 iterations
      if ((i + 1) % 10 == 0) {
        Float64List rewards = await aggregateRollouts(numRollouts: 100, evaluate: true);
        // Save policy weights (simplified to a text file)
        final File file = File('$logdir/lin_policy_plus.txt');
        await file.writeAsString(policy.getWeights().join(','));

        print(params); // Log parameters (simplified)
        print("Time: ${overallStopwatch.elapsed.inSeconds} s");
        print("Iteration: ${i + 1}");
        print("AverageReward: ${rewards.reduce((a, b) => a + b) / rewards.length}");
        print("StdRewards: ${_calculateStdDev(rewards.toList())}");
        print("MaxRewardRollout: ${rewards.reduce(max)}");
        print("MinRewardRollout: ${rewards.reduce(min)}");
        print("Timesteps: $timesteps");
      }

      final syncStopwatch = Stopwatch()..start();
      // Sync all workers' filters with the master filter
      // (The master's filter was updated in aggregateRollouts)
      final filterToSync = policy.observationFilter;
      for (var workerHandle in workers) {
        await workerHandle.syncFilter(filterToSync);
      }

      // Increment stats on all workers
      for (var workerHandle in workers) {
        await workerHandle.statsIncrement();
      }
      syncStopwatch.stop();
      print('Time to sync statistics: ${syncStopwatch.elapsed.inMilliseconds / 1000.0} s');
    }
    overallStopwatch.stop();

    // Clean up isolates
    for (var workerHandle in workers) {
      workerHandle.dispose();
    }
  }

  double _calculateStdDev(List<double> values) {
    if (values.isEmpty) return 0.0;
    final mean = values.reduce((a, b) => a + b) / values.length;
    final variance = values.map((v) => pow(v - mean, 2)).reduce((a, b) => a + b) / values.length;
    return sqrt(variance);
  }
}

/// A helper class to manage an Isolate worker.
class _WorkerHandle {
  late Isolate _isolate;
  late SendPort _sendPort;
  late ReceivePort _receivePort;

  Future<void> spawn({
    required int envSeed,
    required String envName,
    required Map<String, dynamic> policyParams,
    required int noiseTableSeed,
    required int rolloutLength,
    required double deltaStd,
  }) async {
    _receivePort = ReceivePort();
    _isolate = await Isolate.spawn(workerEntryPoint, _receivePort.sendPort);

    // Wait for the worker to send its SendPort back
    _sendPort = await _receivePort.first as SendPort;

    // Send initialization message
    _sendPort.send([
      'init',
      envSeed,
      envName,
      policyParams,
      noiseTableSeed,
      rolloutLength,
      deltaStd,
    ]);

    // Wait for initialization confirmation
    await _receivePort.firstWhere((msg) => msg == 'initialized');
  }

  Future<WorkerResult> doRollouts(
    Float64List policyWeights,
    int numRollouts,
    double shift,
    bool evaluate,
  ) async {
    _sendPort.send(WorkerMessage(
      policyWeights: policyWeights,
      numRollouts: numRollouts,
      shift: shift,
      evaluate: evaluate,
    ));
    return await _receivePort.firstWhere((msg) => msg is WorkerResult) as WorkerResult;
  }

  Future<Map<String, Float64List>> getWeightsPlusStats() async {
    _sendPort.send('get_weights_plus_stats');
    return await _receivePort.firstWhere((msg) => msg is Map) as Map<String, Float64List>;
  }

  Future<void> statsIncrement() async {
    _sendPort.send('stats_increment');
    await _receivePort.firstWhere((msg) => msg == 'stats_incremented');
  }

  Future<void> syncFilter(ObservationFilter filter) async {
    _sendPort.send(filter); // Send the filter object directly
    await _receivePort.firstWhere((msg) => msg == 'filter_synced');
  }

  Future<ObservationFilter> getFilter() async {
    _sendPort.send('get_filter');
    return await _receivePort.firstWhere((msg) => msg is ObservationFilter) as ObservationFilter;
  }

  void dispose() {
    _isolate.kill(priority: Isolate.immediate);
    _receivePort.close();
  }
}

// gym_mock.dart
import 'dart:typed_data';
import 'dart:math';

/// Represents the shape of a space (e.g., observation or action space).
class SpaceShape {
  final List<int> shape;
  SpaceShape(this.shape);
}

/// Represents the result of a step in the environment.
class StepResult {
  final Float64List observation;
  final double reward;
  final bool done;
  final Map<String, dynamic> info;

  StepResult({
    required this.observation,
    required this.reward,
    required this.done,
    this.info = const {},
  });
}

/// Mock specification for an environment.
class EnvSpec {
  final int timestepLimit;
  EnvSpec({this.timestepLimit = 1000});
}

/// Abstract base class for a mock Gym environment.
abstract class GymEnv {
  late SpaceShape observationSpace;
  late SpaceShape actionSpace;
  late EnvSpec spec;

  Float64List reset();
  StepResult step(Float64List action); // Note: For discrete actions, this would typically take an `int`
  void seed(int seed);
  void close();
}

/// A mock implementation of a continuous control environment (e.g., HalfCheetah-v1).
class MockHalfCheetahEnv implements GymEnv {
  @override
  late SpaceShape observationSpace;
  @override
  late SpaceShape actionSpace;
  @override
  late EnvSpec spec;

  final Random _random;
  Float64List _currentObservation = Float64List(17); // Example observation size for HalfCheetah
  int _stepsTaken = 0;

  MockHalfCheetahEnv() : _random = Random() {
    observationSpace = SpaceShape([17]);
    actionSpace = SpaceShape([6]); // Example action size for HalfCheetah
    spec = EnvSpec(timestepLimit: 1000);
  }

  @override
  Float64List reset() {
    _stepsTaken = 0;
    // Generate a random initial observation
    for (int i = 0; i < _currentObservation.length; i++) {
      _currentObservation[i] = _random.nextDouble() * 2 - 1; // Values between -1 and 1
    }
    return _currentObservation;
  }

  @override
  StepResult step(Float64List action) {
    _stepsTaken++;
    // Simulate a reward (e.g., based on action magnitude or just random)
    double reward = action.map((a) => a.abs()).reduce((a, b) => a + b) / action.length * 0.1 + _random.nextDouble() * 0.1;

    // Simulate next observation (e.g., random walk from current)
    for (int i = 0; i < _currentObservation.length; i++) {
      _currentObservation[i] += (_random.nextDouble() * 0.1 - 0.05); // Small random change
    }

    bool done = _stepsTaken >= spec.timestepLimit || _random.nextDouble() < 0.001; // Small chance of early termination

    return StepResult(
      observation: _currentObservation,
      reward: reward,
      done: done,
    );
  }

  @override
  void seed(int seed) {
    // In a real env, this would set the environment's random seed.
    // For this mock, the Random object is already initialized.
    // If you wanted to re-seed, you'd create a new Random(_seed)
  }

  @override
  void close() {
    // In a real env, this would close the environment.
    print('MockHalfCheetahEnv closed.');
  }
}

/// A mock implementation of a Chess environment.
/// NOTE: This is a structural mock. It does NOT implement actual chess game logic.
/// Implementing full chess rules (move generation, legality, check, checkmate, etc.)
/// is a significant undertaking and is beyond the scope of this example.
class MockChessEnv implements GymEnv {
  @override
  late SpaceShape observationSpace;
  @override
  late SpaceShape actionSpace;
  @override
  late EnvSpec spec;

  final Random _random;
  // A simplified observation: e.g., 8x8 board with 12 piece types (6 white, 6 black)
  // plus 4 for castling rights, 1 for en passant, 1 for halfmove clock, 1 for turn.
  // Total: 8*8*12 + 4 + 1 + 1 + 1 = 768 + 7 = 775.
  // This is a common way to flatten a board state for neural networks.
  Float64List _currentObservation = Float64List(775);
  int _stepsTaken = 0;

  MockChessEnv() : _random = Random() {
    // Observation space: 775 features for a flattened board representation
    observationSpace = SpaceShape([775]);
    // Action space: Discrete. Max possible legal moves in chess is around 218.
    // We'll use a large enough integer to represent a move index.
    // A real implementation would map this integer to a specific (from_square, to_square) move.
    actionSpace = SpaceShape([256]); // Example: Max 256 possible discrete moves (e.g., UCT node index)
    spec = EnvSpec(timestepLimit: 200); // Chess games typically shorter than continuous control
  }

  @override
  Float64List reset() {
    _stepsTaken = 0;
    // Simulate an initial chess board state (e.g., starting position)
    // In a real implementation, this would set up the actual board.
    for (int i = 0; i < _currentObservation.length; i++) {
      _currentObservation[i] = _random.nextDouble(); // Dummy values
    }
    return _currentObservation;
  }

  @override
  StepResult step(Float64List action) {
    _stepsTaken++;
    // In a real chess environment, `action` would be a discrete move.
    // Here, we're still taking a Float64List for compatibility with the current Policy.
    // You would need to convert the policy's continuous output to a discrete move.

    // Simulate a reward (very simple for mock: random, or small positive for "progress")
    double reward = _random.nextDouble() * 0.1;

    // Simulate next observation (dummy change)
    for (int i = 0; i < _currentObservation.length; i++) {
      _currentObservation[i] += (_random.nextDouble() * 0.01 - 0.005);
    }

    // Simulate game termination (checkmate, stalemate, draw, or max steps)
    bool done = _stepsTaken >= spec.timestepLimit || _random.nextDouble() < 0.01;

    return StepResult(
      observation: _currentObservation,
      reward: reward,
      done: done,
    );
  }

  @override
  void seed(int seed) {
    // In a real env, this would set the environment's random seed.
  }

  @override
  void close() {
    print('MockChessEnv closed.');
  }
}

// optimizers.dart
import 'dart:typed_data';
import 'package:vector_math/vector_math.dart';

/// Abstract base class for an optimizer.
abstract class Optimizer {
  Float64List _theta;
  double _stepSize;

  Optimizer(this._theta, this._stepSize);

  /// Computes the update step based on the gradient.
  Float64List computeStep(Float64List gradient);
}

/// Stochastic Gradient Descent (SGD) optimizer.
class SGD extends Optimizer {
  SGD(Float64List theta, double stepSize) : super(theta, stepSize);

  @override
  Float64List computeStep(Float64List gradient) {
    // For simple SGD, the step is just -stepSize * gradient
    // In the original Python code, it seems the update is `self.w_policy -= self.optimizer._compute_step(g_hat)`
    // which implies `_compute_step` returns the *update* to subtract.
    // So, it's `stepSize * gradient`.
    return gradient * _stepSize;
  }
}

// policies.dart
import 'dart:typed_data';
import 'dart:math';
import 'package:vector_math/vector_math.dart';

/// Abstract base class for an observation filter.
abstract class ObservationFilter {
  late Float64List mean;
  late Float64List std;
  late int count;
  late Float64List _buffer;
  late int _bufferIdx;
  late int _bufferSize;

  bool get isInitialized;

  /// Updates the filter with a new observation.
  void update(ObservationFilter other);

  /// Filters an observation.
  Float64List filter(Float64List ob, {bool update = true});

  /// Increments statistics.
  void statsIncrement();

  /// Clears the buffer.
  void clearBuffer();

  /// Synchronizes the filter with another filter.
  void sync(ObservationFilter other);
}

/// No filter, returns the observation as is.
class NoFilter implements ObservationFilter {
  @override
  late Float64List mean;
  @override
  late Float64List std;
  @override
  late int count;
  @override
  late Float64List _buffer;
  @override
  late int _bufferIdx;
  @override
  late int _bufferSize;

  NoFilter(int obDim) {
    mean = Float64List(obDim);
    std = Float64List(obDim);
    count = 0;
    _buffer = Float64List(0); // Not used
    _bufferIdx = 0;
    _bufferSize = 0;
  }

  @override
  bool get isInitialized => true;

  @override
  void update(ObservationFilter other) {
    // No-op
  }

  @override
  Float64List filter(Float64List ob, {bool update = true}) {
    return ob;
  }

  @override
  void statsIncrement() {
    // No-op
  }

  @override
  void clearBuffer() {
    // No-op
  }

  @override
  void sync(ObservationFilter other) {
    // No-op
  }
}

/// Mean and Standard Deviation filter.
class MeanStdFilter implements ObservationFilter {
  @override
  late Float64List mean;
  @override
  late Float64List std;
  @override
  late int count;
  @override
  late Float64List _buffer;
  @override
  late int _bufferIdx;
  @override
  late int _bufferSize;

  MeanStdFilter(int obDim, {Float64List? mean, Float64List? std, int? count}) {
    this.mean = mean ?? Float64List(obDim)..fillRange(0, obDim, 0.0);
    this.std = std ?? Float64List(obDim)..fillRange(0, obDim, 1.0);
    this.count = count ?? 0;
    _bufferSize = 1000; // Arbitrary buffer size, can be tuned
    _buffer = Float64List(obDim * _bufferSize);
    _bufferIdx = 0;
  }

  @override
  bool get isInitialized => count > 0;

  @override
  void update(ObservationFilter other) {
    if (other is MeanStdFilter) {
      if (!other.isInitialized) return;

      int newCount = count + other.count;
      if (newCount == 0) return;

      Float64List newMean = Float64List(mean.length);
      Float64List newStd = Float64List(std.length);

      for (int i = 0; i < mean.length; i++) {
        newMean[i] = (mean[i] * count + other.mean[i] * other.count) / newCount;
        // Simplified combined variance calculation (more robust methods exist)
        double var1 = std[i] * std[i];
        double var2 = other.std[i] * other.std[i];
        newStd[i] = sqrt((var1 * count + var2 * other.count) / newCount +
            (count * other.count * pow(mean[i] - other.mean[i], 2)) / (newCount * newCount));
      }
      mean = newMean;
      std = newStd;
      count = newCount;
    }
  }

  @override
  Float64List filter(Float64List ob, {bool update = true}) {
    if (update) {
      if (_bufferIdx < _bufferSize) {
        _buffer.setRange(_bufferIdx * ob.length, (_bufferIdx + 1) * ob.length, ob);
        _bufferIdx++;
      }
    }

    if (!isInitialized) {
      return ob;
    } else {
      Float64List filteredOb = Float64List(ob.length);
      for (int i = 0; i < ob.length; i++) {
        filteredOb[i] = (ob[i] - mean[i]) / (std[i] + 1e-8); // Add epsilon for stability
      }
      return filteredOb;
    }
  }

  @override
  void statsIncrement() {
    if (_bufferIdx > 0) {
      Float64List currentBuffer = Float64List.fromList(_buffer.sublist(0, _bufferIdx * mean.length));
      Float64List newMean = Float64List(mean.length);
      Float64List newStd = Float64List(std.length);
      int newCount = count + _bufferIdx;

      if (newCount == 0) return;

      for (int i = 0; i < mean.length; i++) {
        double sum = 0.0;
        for (int j = 0; j < _bufferIdx; j++) {
          sum += currentBuffer[j * mean.length + i];
        }
        double bufferMean = sum / _bufferIdx;

        newMean[i] = (mean[i] * count + bufferMean * _bufferIdx) / newCount;

        double sumSqDiff = 0.0;
        for (int j = 0; j < _bufferIdx; j++) {
          sumSqDiff += pow(currentBuffer[j * mean.length + i] - bufferMean, 2);
        }
        double bufferVar = sumSqDiff / _bufferIdx;

        double var1 = std[i] * std[i];
        double var2 = bufferVar;

        newStd[i] = sqrt((var1 * count + var2 * _bufferIdx) / newCount +
            (count * _bufferIdx * pow(mean[i] - bufferMean, 2)) / (newCount * newCount));
      }
      mean = newMean;
      std = newStd;
      count = newCount;
    }
  }

  @override
  void clearBuffer() {
    _bufferIdx = 0;
  }

  @override
  void sync(ObservationFilter other) {
    if (other is MeanStdFilter) {
      mean = Float64List.fromList(other.mean);
      std = Float64List.fromList(other.std);
      count = other.count;
    }
  }
}

/// Abstract base class for a policy.
abstract class Policy {
  late Map<String, dynamic> _params;
  late ObservationFilter observationFilter;
  late bool updateFilter;

  Policy(this._params) {
    if (_params['ob_filter'] == 'MeanStdFilter') {
      observationFilter = MeanStdFilter(_params['ob_dim']);
    } else if (_params['ob_filter'] == 'NoFilter') {
      observationFilter = NoFilter(_params['ob_dim']);
    } else {
      throw UnimplementedError('Observation filter ${_params['ob_filter']} not implemented.');
    }
    updateFilter = true;
  }

  /// Acts based on the given observation.
  /// For complex tasks like chess, a simple linear policy is generally insufficient.
  /// A neural network-based policy would be required, which would involve
  /// handling multiple layers, activation functions, and potentially a
  /// more complex mapping from observation to action probabilities.
  Float64List act(Float64List ob);

  /// Updates the policy weights.
  void updateWeights(Float64List newWeights);

  /// Gets the current policy weights.
  Float64List getWeights();
}

/// A linear policy.
class LinearPolicy extends Policy {
  late Float64List _weights; // Flattened weights

  LinearPolicy(Map<String, dynamic> params) : super(params) {
    int obDim = params['ob_dim'];
    int acDim = params['ac_dim'];
    _weights = Float64List(obDim * acDim); // Initialize with zeros
  }

  @override
  Float64List act(Float64List ob) {
    Float64List filteredOb = observationFilter.filter(ob, update: updateFilter);
    int obDim = _params['ob_dim'];
    int acDim = _params['ac_dim'];
    Float64List action = Float64List(acDim);

    // Matrix multiplication: action = filteredOb * _weights (reshaped)
    // Assuming _weights is a (obDim x acDim) matrix, flattened row-major
    for (int i = 0; i < acDim; i++) {
      double sum = 0.0;
      for (int j = 0; j < obDim; j++) {
        sum += filteredOb[j] * _weights[j * acDim + i];
      }
      action[i] = sum;
    }

    return action;
  }

  @override
  void updateWeights(Float64List newWeights) {
    _weights = newWeights;
  }

  @override
  Float64List getWeights() {
    return _weights;
  }

  // This method is not in the original Python policy but is in the worker
  // It's likely meant to return the weights and filter stats together for saving.
  Map<String, dynamic> getWeightsPlusStats() {
    return {
      'weights': _weights,
      'filter_mean': observationFilter.mean,
      'filter_std': observationFilter.std,
      'filter_count': observationFilter.count,
    };
  }
}

// shared_noise.dart
import 'dart:typed_data';
import 'dart:math';
import 'package:vector_math/vector_math.dart';

/// A class to manage a shared noise table.
/// In a truly distributed system, this would be a shared memory segment or a service.
/// Here, it simulates a shared table by generating noise deterministically based on an index.
class SharedNoiseTable {
  final Random _random;
  final Map<int, Float64List> _noiseCache = {};

  SharedNoiseTable.fromSeed(int seed) : _random = Random(seed);

  /// Generates or retrieves a noise vector for a given index and size.
  Tuple2<int, Float64List> getDelta(int size) {
    // Generate a random index for the noise vector
    // In a real shared noise table, this index would be passed around.
    // Here, we just generate a random index to simulate picking from a large pool.
    final int idx = _random.nextInt(1000000); // Arbitrary large number for noise indices

    if (_noiseCache.containsKey(idx)) {
      return Tuple2(idx, _noiseCache[idx]!);
    } else {
      // Generate a new noise vector if not in cache
      final Float64List noise = Float64List(size);
      for (int i = 0; i < size; i++) {
        noise[i] = _random.nextGaussian(); // Standard normal distribution
      }
      _noiseCache[idx] = noise;
      return Tuple2(idx, noise);
    }
  }

  /// Retrieves a noise vector by index.
  Float64List get(int idx, int size) {
    if (_noiseCache.containsKey(idx)) {
      return _noiseCache[idx]!;
    } else {
      // This case should ideally not happen if `getDelta` is always used first.
      // For robustness, regenerate if missing (though this breaks true "shared" concept).
      final Float64List noise = Float64List(size);
      final Random tempRandom = Random(idx); // Use index as seed for reproducibility
      for (int i = 0; i < size; i++) {
        noise[i] = tempRandom.nextGaussian();
      }
      _noiseCache[idx] = noise;
      return noise;
    }
  }
}

// Extension to Random for Gaussian distribution
extension RandomGaussian on Random {
  double nextGaussian({double mean = 0.0, double stdDev = 1.0}) {
    // Box-Muller transform
    double u1 = nextDouble();
    double u2 = nextDouble();
    double randStdNormal = sqrt(-2 * log(u1)) * sin(2 * pi * u2);
    return mean + stdDev * randStdNormal;
  }
}

// Tuple2 class (simple utility, can be replaced by records in Dart 3.0+)
class Tuple2<T1, T2> {
  final T1 item1;
  final T2 item2;
  Tuple2(this.item1, this.item2);
}

// Extension for Float64List to add vector operations
extension Float64ListOperations on Float64List {
  Float64List operator +(Float64List other) {
    if (length != other.length) {
      throw ArgumentError('Vectors must have the same length for addition.');
    }
    final result = Float64List(length);
    for (int i = 0; i < length; i++) {
      result[i] = this[i] + other[i];
    }
    return result;
  }

  Float64List operator -(Float64List other) {
    if (length != other.length) {
      throw ArgumentError('Vectors must have the same length for subtraction.');
    }
    final result = Float64List(length);
    for (int i = 0; i < length; i++) {
      result[i] = this[i] - other[i];
    }
    return result;
  }

  Float64List operator *(double scalar) {
    final result = Float64List(length);
    for (int i = 0; i < length; i++) {
      result[i] = this[i] * scalar;
    }
    return result;
  }

  Float64List operator /(double scalar) {
    if (scalar == 0) {
      throw ArgumentError('Division by zero.');
    }
    final result = Float64List(length);
    for (int i = 0; i < length; i++) {
      result[i] = this[i] / scalar;
    }
    return result;
  }

  double norm() {
    double sumSq = 0.0;
    for (int i = 0; i < length; i++) {
      sumSq += this[i] * this[i];
    }
    return sqrt(sumSq);
  }
}
