from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    overall = summary[summary["scope"] == "overall"].copy()
    columns = overall.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in overall.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    labels = overall["backend"].tolist()
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        ("inference_mean_ms", "Inference mean (ms)", "#4C78A8"),
        ("end_to_end_p50_ms", "End-to-end P50 (ms)", "#F58518"),
        ("end_to_end_p95_ms", "End-to-end P95 (ms)", "#E45756"),
        ("end_to_end_throughput_img_s", "End-to-end throughput (img/s)", "#54A24B"),
        ("gpu_memory_peak_delta_mib", "Process GPU memory peak delta (MiB)", "#B279A2"),
    ]
    for axis, (column, title, color) in zip(
        axes.flat[:len(panels)], panels, strict=True
    ):
        values = overall[column].tolist()
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        axis.bar_label(bars, fmt="%.2f", padding=3)
    axes.flat[-1].set_visible(False)
    figure.suptitle("YOLO inference backend comparison")
    figure.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(figure)
