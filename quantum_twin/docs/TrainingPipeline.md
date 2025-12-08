# Training Pipeline

Flow: **data → model → loss → trainer → checkpoint → evaluation → export**

1. `QuantumTrajectoryLoader` builds batches from `DataSimulator`.
2. `ModelFactory` constructs the `PINNModel` (layers, activation, Fourier features).
3. `LossFactory` produces physics losses (`SchrodingerLoss` or `LindbladLoss`) plus regularization.
4. `Trainer`:
   - Computes data MSE + physics residuals.
   - Logs losses/metrics to TensorBoard.
   - Steps optimizer and optional scheduler.
   - Saves checkpoints via `CheckpointManager`.
5. Metrics (`FidelityMetric`, `TraceDistanceMetric`, `PositivityMetric`) run periodically.
6. After training, `ONNXExporter` writes an ONNX model for deployment.

Configuration is declared in `quantum_twin/configs/training.yml` using step-based class loading.

