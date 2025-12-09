# Quantum Twin (PINN-based Qubit Surrogate)

Quantum Twin is a modular, Java-style Python codebase that builds physics-informed neural network (PINN) surrogates for qubit dynamics. It simulates trajectories (Schrödinger + Lindblad), trains a PINN with physics losses, exports ONNX, and serves fast inference with ONNXRuntime. The repository uses step-based YAML configs, Google-style docstrings, and consistent logging across all classes.

## Features
- Physics-informed neural network for qubit state evolution (Schrödinger + Lindblad)
- Step-based YAML configuration with dynamic class loading
- Data simulation with decoherence (T1, T2, Tphi), drift, and pulse sampling
- Logging in every component (constructors, lifecycle, checkpoints, errors)
- TensorBoard integration, checkpoints, and metrics (fidelity, trace distance, positivity)
- ONNX export plus ONNXRuntime inference server and lightweight wrapper

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Quickstart
1) **Install**
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```
2) **Train with step-based config**
```bash
python -m quantum_twin.train --config quantum_twin/configs/training.yml
```
3) **View TensorBoard**
```bash
tensorboard --logdir runs
```
4) **Simulate only**
```bash
python -m quantum_twin.simulate --config quantum_twin/configs/simulation.yml
```
5) **Optimise controls/parameters**
```bash
python -m quantum_twin.optimise --config quantum_twin/configs/optimiser.yml
```
6) **Run demos (persona flows)**
```bash
# Physicist / simulation-focused
python -m quantum_twin.demos.physicist_demo.PhysicistDemo

# Control calibration
python -m quantum_twin.demos.control_demo.ControlCalibrationDemo

# Stakeholder dashboard
python -m quantum_twin.demos.stakeholder_demo.StakeholderDashboardDemo

# Algorithm end-user demo
python -m quantum_twin.demos.algorithm_demo.AlgorithmEndUserDemo
```

7) **Run API orchestrator (pydash-style high-level access)**
```bash
python -c "from quantum_twin.api.QuantumTwinAPI import QuantumTwinAPI; twin = QuantumTwinAPI('quantum_twin/configs/api.yml'); print(twin.simulator.simulate_lindblad(1)['rho'].shape)"
```

8) **Serve ONNX**
```bash
python - <<'PY'
from quantum_twin.deployment.ModelWrapper import ModelWrapper
wrapper = ModelWrapper("artifacts/pinn.onnx")
print(wrapper.run(t=0.05, controls=[0.1, -0.1, 0.0]))
PY
```

9) **Generate plots with pydash_plotting**
```bash
python - <<'PY'
import numpy as np
from quantum_twin.plotting.PlotManager import PlotManager

pm = PlotManager("quantum_twin/configs/plotting.yml")
dummy = {
    "SIMULATION_PLOTS": {"rho": np.zeros((5,2,2),dtype=np.complex128), "t": np.linspace(0,1,5), "controls": np.zeros((5,3))},
    "SURROGATE_PLOTS": {"rho_true": np.zeros((5,2,2),dtype=np.complex128), "rho_pred": np.zeros((5,2,2),dtype=np.complex128), "t": np.linspace(0,1,5)},
}
pm.render_all(dummy)
PY
```

10) **Interactive plots via Dash web app**
```bash
# Option A: static gallery (if dash installed)
python -m quantum_twin.demos.dash_app --outputs outputs --port 8050

# Option B: interactive Plotly Dash (hover/click on points)
python -m quantum_twin.demos.dash_interactive
# Open http://localhost:8050 to interactively explore Bloch trajectory, populations, heatmap
```

## Config Structure (step format)
Each YAML file lists steps with `class` + `params`:
```yaml
component_name:
  class: full.module.ClassName
  params:
    key: value
```
- `quantum_twin/configs/training.yml`: steps for dataloader, model, loss, metrics, training_params.
- `quantum_twin/configs/simulation.yml`: simulator/export/metrics steps.
- `quantum_twin/configs/optimiser.yml`: parameter/control/surrogate steps.

## Repository Layout
- `quantum_twin/core`: logging, config loader, base component, constants
- `quantum_twin/dataloader`: base loader and trajectory loader
- `quantum_twin/physics`: Hamiltonian, pulse generator, Lindblad operators, solvers, simulator
- `quantum_twin/losses`: physics losses and regularization
- `quantum_twin/models`: PINN model, Fourier features, factory
- `quantum_twin/training`: trainer, schedulers, checkpoints, TensorBoard
- `quantum_twin/metrics`: fidelity, trace distance, positivity
- `quantum_twin/deployment`: ONNX exporter, runtime server, wrapper
- `quantum_twin/optimisation`: parameter/pulse/surrogate optimizers (uniform/gaussian/gradient control variants)
- `quantum_twin/docs`: conceptual, physics, architecture, simulation, training, optimization, deployment guides

## Development Notes
- Python 3.10+, fully typed, Google-style docstrings, PEP8
- All classes inherit `BaseComponent` for uniform logging
- Extend via new config steps; swap loss/model/simulator without changing code
- Generated artifacts/datasets are stored under `artifacts/` (gitignored); use `output_path` in simulator config and `input_path` in training dataloader to reuse datasets.
