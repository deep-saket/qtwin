from __future__ import annotations

import numpy as np
import torch
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from quantum_twin.physics.DataSimulator import DataSimulator


class InteractiveDemo:
    def __init__(self, sim_params: dict | None = None) -> None:
        self.sim_params = sim_params or {
            "t_max": 1.0,
            "steps": 200,
            "t1": 30.0,
            "t2": 20.0,
            "tphi": 0.01,
            "drift": 0.02,
            "pulse_scale": 0.5,
            "use_lindblad": True,
        }
        self.simulator = DataSimulator(**self.sim_params)
        self.data = self.simulator.generate_dataset(trajectories=1, use_lindblad=self.sim_params.get("use_lindblad", True))
        self.t = self.data["t"].cpu().numpy().flatten()
        self.rho = self.data["rho"].cpu().numpy()

    def bloch_vectors(self) -> np.ndarray:
        vecs = []
        for r in self.rho:
            sx = 2 * np.real(r[0, 1])
            sy = -2 * np.imag(r[0, 1])
            sz = np.real(r[0, 0] - r[1, 1])
            vecs.append([sx, sy, sz])
        return np.array(vecs)

    def population(self) -> tuple[np.ndarray, np.ndarray]:
        p0 = np.real(self.rho[:, 0, 0])
        p1 = np.real(self.rho[:, 1, 1])
        return p0, p1

    def residuals(self) -> np.ndarray:
        return np.real(self.rho[0])


def build_app(demo: InteractiveDemo) -> dash.Dash:
    app = dash.Dash(__name__)
    vecs = demo.bloch_vectors()
    p0, p1 = demo.population()

    app.layout = html.Div(
        [
            html.H2("Quantum Twin Interactive Demo"),
            dcc.Graph(
                id="bloch-graph",
                figure=go.Figure(
                    data=[go.Scatter3d(x=vecs[:, 0], y=vecs[:, 1], z=vecs[:, 2], mode="lines+markers", text=[f"t={t:.3f}" for t in demo.t])],
                    layout=go.Layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), title="Bloch Trajectory"),
                ),
            ),
            dcc.Graph(
                id="population-graph",
                figure=go.Figure(
                    data=[
                        go.Scatter(x=demo.t, y=p0, name="P(|0>)"),
                        go.Scatter(x=demo.t, y=p1, name="P(|1>)"),
                    ],
                    layout=go.Layout(title="Populations vs Time", xaxis_title="t", yaxis_title="Population"),
                ),
            ),
            dcc.Graph(
                id="residual-heatmap",
                figure=go.Figure(
                    data=go.Heatmap(z=demo.residuals(), colorscale="Magma"),
                    layout=go.Layout(title="Density Snapshot (Real part)", xaxis_title="Col", yaxis_title="Row"),
                ),
            ),
            html.Div("Hover over points to inspect values; reload page after rerunning demos to refresh outputs."),
        ],
        style={"fontFamily": "Arial", "padding": "1rem"},
    )
    return app


def main() -> None:
    demo = InteractiveDemo()
    app = build_app(demo)
    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
