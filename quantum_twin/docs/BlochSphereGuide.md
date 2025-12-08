# Bloch Sphere Guide

The Bloch sphere represents a single qubit state \\(\\rho\\) via a vector \\(\\vec{r} = (x,y,z)\\).

## Conversion
For density matrix \\(\\rho\\):
- \\(x = 2\\Re(\\rho_{01})\\)
- \\(y = -2\\Im(\\rho_{01})\\)
- \\(z = \\rho_{00} - \\rho_{11}\\)

`BlochSphereUtils.density_to_bloch` converts 2x2 density matrices to vectors.

## Plotting
- `BlochSpherePlot` draws trajectories on a 3D Bloch sphere.
- Input: array of density matrices `rho`.
- Optional: animate by saving frames from successive calls to `render`.

## Tips
- Ensure states are physical (Hermitian, PSD, trace-1) before plotting.
- Use subsampling for long trajectories to keep plots light.
