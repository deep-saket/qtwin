# Data Simulation

This module synthesizes qubit trajectories for training the PINN surrogate.

## Components
- `PulseGenerator`: samples control pulses \\((\Omega_x, \Omega_y, \Omega_z)\\).
- `Hamiltonian`: builds \\(H(t) = \text{drift}\,Z + \sum_i c_i \sigma_i\\).
- `SchrodingerSolver`: integrates \\(\\partial_t \\psi = -i H \\psi\\).
- `LindbladSolver`: integrates \\(\\partial_t \\rho = -i[H,\\rho] + \\sum_k L_k \\rho L_k^\\dagger - \\frac{1}{2}\\{L_k^\\dagger L_k, \\rho\\}\\).
- `DataSimulator`: orchestrates random control sampling and solver selection, returns torch tensors.

## ODE Integration
- Uses `scipy.integrate.solve_ivp` with RK45 on vectorized states.
- Supports both pure-state Schrödinger and density-matrix Lindblad dynamics.

## Noise & Decoherence
- Relaxation (`T1`) and dephasing (`T2`, `Tphi`) set collapse operator strengths.
- Drift term adds unwanted rotations to the Hamiltonian.

## Pulse Generation
- Uniform random pulses in a configurable range.
- Optional sinusoidal pulses via `PulseGenerator.sinusoidal`.

## Dataset Output
- Returns tensors: time `t`, controls `controls`, density matrices `rho`.
- Compatible with `QuantumTrajectoryLoader` for batching and device transfer.

