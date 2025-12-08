from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.Constants import DEFAULT_ONNX_PATH
from quantum_twin.deployment.ONNXExporter import ONNXExporter
from quantum_twin.metrics.MetricBase import MetricBase
from quantum_twin.training.CheckpointManager import CheckpointManager
from quantum_twin.training.SchedulerFactory import SchedulerFactory
from quantum_twin.training.TensorboardWriter import TensorboardWriter


class Trainer(BaseComponent):
    """Coordinates model training with physics and data losses."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        dataloader: Iterable[Tuple[torch.Tensor, ...]],
        metrics: List[MetricBase],
        training_params: Dict[str, float],
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.metrics = metrics
        self.params = training_params
        self.logger.info("Trainer initialized with params %s", training_params)

        lr = float(self.params.get("learning_rate", 1e-3))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = SchedulerFactory().build(
            self.optimizer, self.params.get("scheduler"), **self.params.get("scheduler_params", {})
        )
        self.checkpoints = CheckpointManager(
            directory=self.params.get("checkpoint_dir", "artifacts/checkpoints"),
            keep_last=int(self.params.get("keep_last", 3)),
        )
        self.tb = TensorboardWriter(self.params.get("tensorboard_dir", "runs"))
        self.exporter = ONNXExporter(
            export_path=self.params.get("onnx_path", DEFAULT_ONNX_PATH),
            opset=int(self.params.get("onnx_opset", 17)),
        )

    def _compute_losses(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        t, controls, rho_true = batch
        rho_pred = self.model(t, controls)
        data_loss = torch.mean(torch.abs(rho_pred - rho_true) ** 2)
        physics_loss = self.loss_fn.compute(self.model, {"t": t, "controls": controls})
        total_loss = data_loss + physics_loss
        return {"total": total_loss, "data": data_loss, "physics": physics_loss}

    def _log_metrics(self, step: int, rho_pred: torch.Tensor, rho_true: torch.Tensor) -> None:
        for metric in self.metrics:
            values = metric(rho_pred, rho_true)
            for name, val in values.items():
                self.tb._writer.add_scalar(f"metrics/{name}", val, step)

    def train(self) -> None:
        steps = int(self.params.get("training_steps", 1000))
        log_interval = int(self.params.get("log_interval", 50))
        ckpt_interval = int(self.params.get("checkpoint_interval", 200))

        for step, batch in zip(range(steps), self.dataloader):
            t, controls, rho_true = batch
            t = t.double()
            controls = controls.double()
            rho_true = rho_true.to(torch.cdouble)

            self.optimizer.zero_grad()
            losses = self._compute_losses((t, controls, rho_true))
            losses["total"].backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if step % log_interval == 0:
                self.logger.info(
                    "Step %d total=%.6f data=%.6f physics=%.6f",
                    step,
                    losses["total"].item(),
                    losses["data"].item(),
                    losses["physics"].item(),
                )
                self.tb.log_scalars(
                    step,
                    {
                        "loss/total": float(losses["total"].item()),
                        "loss/data": float(losses["data"].item()),
                        "loss/physics": float(losses["physics"].item()),
                    },
                )
                self._log_metrics(step, self.model(t, controls), rho_true)

            if step % ckpt_interval == 0 and step > 0:
                self.checkpoints.save(step, self.model, self.optimizer)

        sample_batch = next(iter(self.dataloader))
        export_input = {"t": sample_batch[0].double(), "controls": sample_batch[1].double()}
        self.exporter.export(self.model, export_input)
        self.tb.close()
        self.logger.info("Training complete")

