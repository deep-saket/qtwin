# Plotting Overview

The `pydash_plotting` subsystem provides configurable visualizations for simulation, surrogate training, optimisation, what-if analysis, training diagnostics, and deployment benchmarks. All plotters inherit from `BasePlot` and use the project logging system.

## How It Works
- Configuration: `configs/plotting.yml` declares plot groups (simulation, surrogate, optimisation, training, what-if, deployment).
- Factory: `PlotFactory` dynamically instantiates plot classes using `class` and `params`.
- Manager: `PlotManager` loads the config, builds the requested plots, and renders them with datasets passed at runtime.

## Usage
```python
from quantum_twin.plotting.PlotManager import PlotManager

manager = PlotManager("quantum_twin/configs/plotting.yml")
datasets = {
    "SIMULATION_PLOTS": {"rho": rho_sim, "t": t_sim, "controls": controls},
    "SURROGATE_PLOTS": {"rho_true": rho_true, "rho_pred": rho_pred, "t": t_train},
}
manager.render_all(datasets)
```
