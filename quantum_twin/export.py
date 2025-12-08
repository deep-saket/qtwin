from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import torch

from quantum_twin.core.ConfigLoader import ConfigLoader
from quantum_twin.core.StepInstantiator import StepInstantiator
from quantum_twin.training.Trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PINN model to ONNX")
    parser.add_argument("--config", type=str, default="quantum_twin/configs/training.yml", help="Training config path")
    args = parser.parse_args()

    loader = ConfigLoader(args.config)
    raw = loader.load_yaml()
    steps = loader.parse_steps(raw)
    step_map = {step.name: step for step in steps}
    inst = StepInstantiator()

    dataloader = inst.instantiate(step_map["dataloader"])
    model_factory = inst.instantiate(step_map["model"])
    loss_factory = inst.instantiate(step_map["loss"])
    metrics_factory = inst.instantiate(step_map["metrics"])
    training_params = step_map["training_params"].params

    model = model_factory.build()
    loss_fn = loss_factory.build()
    metrics = metrics_factory.build()

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        dataloader=dataloader.loader(),
        metrics=metrics,
        training_params=training_params,
    )

    # Run one forward pass to build export inputs
    sample_batch = next(iter(dataloader.loader()))
    param_device = next(model.parameters()).device
    export_input = {
        "t": sample_batch[0].to(dtype=next(model.parameters()).dtype),
        "controls": sample_batch[1].to(dtype=next(model.parameters()).dtype),
    }
    trainer.exporter.export(model, export_input)


if __name__ == "__main__":
    main()
