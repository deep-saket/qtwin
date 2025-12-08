# Plot Catalog

## Simulation
- `DensityMatrixPlot`: real/imag heatmaps of ρ(t) snapshots.
- `BlochSpherePlot`: Bloch vector trajectory.
- `PulsePlot`: control pulses Ωx/Ωy/Ωz.
- `PopulationPlot`: P(|0⟩) and P(|1⟩) over time.

## Surrogate
- `PredictionPlot`: ground truth vs predicted density entries.
- `ErrorPlot`: error trajectory.
- `ResidualPlot`: physics residual heatmap.
- `EigenvalueTrajectoryPlot`: eigenvalue tracks for positivity checks.
- `LossCurvePlot`: training/validation/physics vs supervised loss.
- `HeatmapPlot`: general-purpose comparison heatmap.

## Optimisation
- `ParameterEstimationPlot`: convergence vs true parameters.
- `ControlPulseOptimPlot`: pulse updates across iterations.
- `CostLandscapePlot`: 2D cost contours.
- `SurrogateFitComparisonPlot`: surrogate vs simulator/experiment.

## What-if
- `NoiseScanPlot`: fidelity/error vs T1/T2 sweeps.
- `PulseSweepPlot`: amplitude/frequency sweeps.
- `SensitivityPlot`: ∂ρ/∂param heatmaps.
- `ScenarioComparisonRadarPlot`: multi-scenario radar.

## Training
- `TrainingCurvePlot`: loss and gradient norms.
- `CheckpointDriftPlot`: checkpoint comparison.

## Deployment
- `ONNXBenchmarksPlot`: latency/throughput.
- `DriftOverServingPlot`: error drift across calls.
