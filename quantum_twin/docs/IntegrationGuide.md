# Integration Guide

## Training
- After each epoch or checkpoint, call `PlotManager.render_group("TRAINING_PLOTS", loss=loss_history, grad_norm=grad_norms)`.
- For physics/surrogate diagnostics, use `SURROGATE_PLOTS` with `rho_true`, `rho_pred`, `residuals`, `train_loss`, `val_loss`.

## Simulation
- From `DataSimulator`, feed `rho`, `t`, and `controls` into `SIMULATION_PLOTS`.

## Optimisation
- For parameter estimation, call `OPTIMISATION_PLOTS` with `estimates`, `true_params`.
- For control pulses, pass `controls_history` arrays.

## Deployment
- Benchmark ONNX inference and feed `latency_ms`, `throughput` to `DEPLOYMENT_PLOTS`.

## What-if
- Sweep parameters (T1/T2, pulse amplitude, etc.) and pass metrics to `WHATIF_PLOTS`.

## Saving
- Each plot uses `save_path` from config. Override by calling `plotter.save(path)` after `render`.

## Logging
- All plotters inherit `BaseComponent`, so rendering/saving is logged with class-specific loggers.
