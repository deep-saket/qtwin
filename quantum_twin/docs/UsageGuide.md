# Usage Guide: Quantum Twin

## 1. Audience
- **ML engineers:** Train PINN surrogates, monitor losses/metrics, export ONNX for low-latency inference.
- **Physicists:** Simulate qubit dynamics, inspect density matrices/Bloch trajectories, validate physics losses.
- **Control/calibration engineers:** Optimize pulses/parameters, run what-if sweeps, compare checkpoints.
- **Stakeholders:** Consume dashboards/plots to assess fidelity, latency, and robustness.

## 2. Running the Simulator
- Configure `quantum_twin/configs/simulation.yml` (class + params for simulator/export/metrics).
- Use `DataSimulator` directly or via `PlotManager`:
```python
from quantum_twin.physics.DataSimulator import DataSimulator
sim = DataSimulator({"t_max":1.0, "steps":200, "t1":30.0, "t2":20.0, "tphi":0.01, "pulse_scale":0.5})
data = sim.generate_dataset(trajectories=4, use_lindblad=True)
```
- Visualize with plotting config:
```python
from quantum_twin.plotting.PlotManager import PlotManager
pm = PlotManager("quantum_twin/configs/plotting.yml")
pm.render_group("SIMULATION_PLOTS", rho=data["rho"], t=data["t"], controls=data["controls"])
```

## 3. Training the Surrogate (`train.py`)
- Config: `quantum_twin/configs/training.yml` has 5 steps (dataloader, model, loss, metrics, training_params) each with `class` and `params`.
- How it works:
  - `ConfigLoader.parse_steps` reads YAML → `StepInstantiator` builds objects.
  - `QuantumTrajectoryLoader` produces batches from `DataSimulator`.
  - `ModelFactory` builds `PINNModel` (Fourier features, dropout, Cholesky density output).
  - `LossFactory` builds physics loss (Schrödinger/Lindblad + regularization).
  - `Trainer` runs loops, combines data+physics loss, logs TensorBoard, saves checkpoints, exports ONNX at end.
- Run:
```bash
python -m quantum_twin.train --config quantum_twin/configs/training.yml
tensorboard --logdir runs  # monitor loss/metrics
```
- Checkpoints: saved under `artifacts/checkpoints/step_*.pt`; auto-pruned by `CheckpointManager`.

## 4. Optimization Workflows
- Config: `quantum_twin/configs/optimiser.yml` (parameter_estimator, control_optimizer, surrogate_fitter).
- Tools:
  - `ParameterEstimator`: heuristics/estimation of drift/parameters.
  - `ControlOptimizer`: choose uniform/gaussian/gradient classes for pulse search.
  - `SurrogateFitter`: align surrogate to simulator/experiment.
- Run:
```bash
python -m quantum_twin.optimise --config quantum_twin/configs/optimiser.yml
```
- Surrogate speeds up inner-loop evaluations vs full ODE solves.

## 5. What-If Analysis
- Sweep noise/pulses/sensitivities via `PlotManager` groups in `plotting.yml` (`WHATIF_PLOTS`).
- Example: generate metrics for varying T1/T2, then render:
```python
pm.render_group("WHATIF_PLOTS", t1=t1_values, metric=fidelity_values,
                amplitude=pulse_amplitudes, sensitivity=sensitivity_map,
                scenarios=scenario_matrix, labels=scenario_labels)
```
- Use `ScenarioComparisonRadarPlot` for multi-scenario summaries.

## 6. Deployment & Inference
- ONNX export: auto at end of training via `ONNXExporter` (default `artifacts/pinn.onnx`).
- Inference:
```python
from quantum_twin.deployment.ModelWrapper import ModelWrapper
mw = ModelWrapper("artifacts/pinn.onnx")
rho = mw.run(t=0.05, controls=[0.1, -0.1, 0.0])
```
- Serving: `ONNXRuntimeServer` for batch inference; feed NumPy arrays for `t` and `controls`.
- Integrate into larger systems by wrapping `ModelWrapper` inside service endpoints or control loops; monitor with `DEPLOYMENT_PLOTS` (latency/drift).
