# Architecture: Quantum Twin

## 1. Directory Structure
- `core/`: logging (`BaseLogger`, `LoggerFactory`), base component (`BaseComponent`), config loader/instantiator.
- `physics/`: Hamiltonian, Lindblad operators, solvers (Schrodinger/Lindblad), data simulator.
- `models/`: PINN model, Fourier features, model factory.
- `training/`: Trainer, schedulers, checkpoints, TensorBoard writer.
- `dataloader/`: dataset + loader for trajectories.
- `losses/`: physics-informed losses (Schrödinger, Lindblad, regularization).
- `metrics/`: fidelity, trace distance, positivity metrics and factory.
- `optimisation/`: parameter estimators, control optimizers, surrogate fitter.
- `deployment/`: ONNX exporter, ONNXRuntime server, model wrapper.
- `plotting/`: pydash_plotting subsystem for simulation/surrogate/optimisation/what-if/training/deployment plots.
- `configs/`: YAML step configs (class + params) for training, simulation, optimiser, plotting.
- `docs/`: conceptual, architecture, plotting, usage docs.

## 2. OOP Patterns
- **Java-like inheritance:** Every class derives from `BaseComponent` → `BaseLogger` for consistent logging (`self.logger`, `cls.logger`).
- **Factories:** `ModelFactory`, `LossFactory`, `MetricsFactory`, `PlotFactory` dynamically import classes from config.
- **Config loader:** Step-based YAML with `class` and `params`; `ConfigLoader.parse_steps` + `StepInstantiator` build objects.
- **Separation:** Physics, models, losses, training, optimisation, plotting, deployment each in dedicated packages.

## 3. Data Flow
```
Simulation (physics.DataSimulator)
    └─> Dataloader (QuantumTrajectoryLoader)
         └─> Training (Trainer + PINNModel + PhysicsLoss)
              └─> Surrogate (ONNX export)
                   └─> Optimisation (Parameter/Control/Surrogate fit)
                        └─> Deployment (ONNXRuntimeServer / ModelWrapper)
```

Config-driven instantiation:
```
YAML (class + params)
    └─ ConfigLoader.parse_steps
        └─ StepInstantiator.instantiate
            └─ Concrete object
```

## 4. Components
- **Physics solvers:** `SchrodingerSolver`, `LindbladSolver`, `DataSimulator` generate trajectories with drift, decoherence, pulses.
- **PINN model:** `PINNModel` (Cholesky density output, Fourier features); built via `ModelFactory`.
- **Physics losses:** `SchrodingerLoss`, `LindbladLoss`, `RegularizationLoss` enforce residuals + trace/hermiticity/positivity.
- **Optimizers:** `OptimizerBase`; control optimizers (uniform/gaussian/gradient), `ParameterEstimator`, `SurrogateFitter`.
- **Plotting:** `PlotFactory`, `PlotManager`, plot classes by domain (simulation, surrogate, optimisation, what-if, training, deployment).
- **Training pipeline:** `Trainer` combines data + physics loss, metrics, checkpoints, TensorBoard logging.
- **Deployment:** `ONNXExporter` -> `ONNXRuntimeServer` -> `ModelWrapper` for fast inference.

## 5. Checkpointing & TensorBoard
- **Checkpoints:** `CheckpointManager` saves `{model, optimizer, step}` to `artifacts/checkpoints/step_*.pt`, prunes old checkpoints.
- **TensorBoard:** `TensorboardWriter` logs losses/metrics, histograms/images; `Trainer` calls it each log interval.

## 6. Deployment Flow
1) Train PINN → `Trainer` exports ONNX via `ONNXExporter` to `artifacts/pinn.onnx`.
2) Serve with `ONNXRuntimeServer` (CPUExecutionProvider by default).
3) Wrap inference via `ModelWrapper` for simple `run(t, controls)` calls.

### ASCII Component Diagram
```
Configs (YAML class+params)
   |
   v
ConfigLoader + StepInstantiator
   |
   +--> DataSimulator --> QuantumTrajectoryLoader
   +--> ModelFactory --> PINNModel
   +--> LossFactory  --> PhysicsLoss
   +--> MetricsFactory --> Metrics
   |
Trainer --> TensorboardWriter --> CheckpointManager --> ONNXExporter
   |
   +--> PlotManager (simulation/surrogate/training/what-if/deployment)
   |
ONNXRuntimeServer / ModelWrapper
```
