# Optimization

The optimisation module provides reusable abstractions:

- `OptimizerBase`: abstract interface with `run()`.
- `ParameterEstimator`: estimates Hamiltonian/drift parameters from data.
- `ControlOptimizer`: proposes control pulses for target objectives.
- `SurrogateFitter`: fits a surrogate to a more expensive simulator.

These tools can wrap advanced algorithms (gradient-based or evolutionary) to tune Hamiltonians, noise rates, or pulse schedules while reusing the logging and configuration system.

