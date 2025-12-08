# Physics Background

## Complex Vector Space
- Qubit states live in \\(\\mathbb{C}^2\\).
- Inner product \\(\\langle \\phi | \\psi \\rangle\\) defines probabilities; unitary evolution preserves norms.

## Pauli Matrices
\\[
\\sigma_x = \\begin{bmatrix}0 & 1\\\\1 & 0\\end{bmatrix},\\;
\\sigma_y = \\begin{bmatrix}0 & -i\\\\i & 0\\end{bmatrix},\\;
\\sigma_z = \\begin{bmatrix}1 & 0\\\\0 & -1\\end{bmatrix}
\\]

## Schrödinger Equation
- Describes closed-system evolution: \\(\\partial_t |\\psi\\rangle = -i H |\\psi\\rangle\\).
- Density form: \\(\\partial_t \\rho = -i[H,\\rho]\\).

## Lindblad Master Equation
- Open-system evolution with collapse operators \\(L_k\\):
\\[
\\partial_t \\rho = -i[H,\\rho] + \\sum_k L_k \\rho L_k^\\dagger - \\tfrac{1}{2}\\{L_k^\\dagger L_k, \\rho\\}
\\]
- T1 relaxation models energy decay; T2 covers dephasing; extra pure-dephasing captured by \\(\\sqrt{\\gamma_\\phi}\\sigma_z\\).

## Noise and Dissipation
- **Relaxation (T1):** population decays to ground state.
- **Dephasing (T2):** phases randomize, shrinking Bloch vector toward z-axis.
- **Drive/Drift:** control pulses drive rotations; drift offsets add unwanted rotations.

## Why Surrogates?
- Full simulation can be costly for repeated evaluations in calibration or control loops.
- PINN surrogate offers fast evaluation while respecting the governing equations.

