from __future__ import annotations

import base64
from pathlib import Path
from typing import List

from threading import Thread

try:
    import dash
    from dash import html, dcc
    from dash.dependencies import Input, Output
except ModuleNotFoundError:
    dash = None  # type: ignore
    html = dcc = Input = Output = None  # type: ignore


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in folder.glob("**/*") if p.suffix.lower() in exts)


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _build_app(outputs_dir: str) -> dash.Dash:
    if dash is None:
        raise RuntimeError("Dash is not installed. Install 'dash' to use the web viewer.")
    folder = Path(outputs_dir)
    folder.mkdir(parents=True, exist_ok=True)

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Quantum Twin Demo Plots"),
            dcc.Interval(id="refresh", interval=5000, n_intervals=0),
            html.Div(id="gallery"),
        ],
        style={"fontFamily": "Arial", "padding": "1rem"},
    )

    @app.callback(Output("gallery", "children"), [Input("refresh", "n_intervals")])
    def refresh(_n: int) -> List[html.Div]:
        cards: List[html.Div] = []
        for img in list_images(folder):
            cards.append(
                html.Div(
                    [
                        html.H4(img.name),
                        html.Img(src=encode_image(img), style={"maxWidth": "600px"}),
                    ],
                    style={"marginBottom": "1rem"},
                )
            )
        if not cards:
            cards = [html.Div("No images found yet. Generate plots under outputs/.")]
        return cards

    return app


def serve_gallery(outputs_dir: str = "outputs", port: int = 8050) -> None:
    if dash is None:
        print("Dash is not installed; skipping web viewer.")
        return
    app = _build_app(outputs_dir)
    app.run(debug=False, host="0.0.0.0", port=port)


def launch_dash_async(outputs_dir: str = "outputs", port: int = 8050) -> Thread:
    if dash is None:
        print("Dash is not installed; skipping web viewer.")
        return None  # type: ignore
    app = _build_app(outputs_dir)
    thread = Thread(
        target=app.run,
        kwargs={"debug": False, "host": "0.0.0.0", "port": port, "use_reloader": False},
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    serve_gallery()
