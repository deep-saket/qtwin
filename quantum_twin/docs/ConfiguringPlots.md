# Configuring Plots

`configs/plotting.yml` declares plot groups using `class` and `params` entries.

Example:
```yaml
SIMULATION_PLOTS:
  - class: quantum_twin.plotting.simulation.DensityMatrixPlot.DensityMatrixPlot
    params:
      save_path: outputs/density_matrix.png
      cmap: viridis
```

Groups:
- `SIMULATION_PLOTS`
- `SURROGATE_PLOTS`
- `OPTIMISATION_PLOTS`
- `TRAINING_PLOTS`
- `WHATIF_PLOTS`
- `DEPLOYMENT_PLOTS`

At runtime, pass data per group to `PlotManager.render_all`.

Minimal code:
```python
manager = PlotManager("quantum_twin/configs/plotting.yml")
manager.render_group("SIMULATION_PLOTS", rho=rho, t=t, controls=controls)
```
