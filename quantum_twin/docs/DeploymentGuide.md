# Deployment Guide

## Export to ONNX
- Training concludes with `ONNXExporter.export`, producing `artifacts/pinn.onnx` (configurable).
- Inputs: `t` (batch,1) and `controls` (batch,3). Output: `rho` (batch,2,2).

## ONNXRuntime Inference
```python
from quantum_twin.deployment.ModelWrapper import ModelWrapper
wrapper = ModelWrapper("artifacts/pinn.onnx")
rho = wrapper.run(t=0.05, controls=[0.1, -0.1, 0.0])
print(rho)
```

## Server Usage
- `ONNXRuntimeServer` loads the ONNX model and exposes `predict` for batched NumPy inputs.
- Swap providers or devices by editing the server initialization.

## Tips
- Keep the ONNX opset aligned with target runtimes (default 17).
- Ensure training/export uses double precision inputs for consistency with physics losses.
