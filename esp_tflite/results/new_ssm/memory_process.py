from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


METRIC_PATTERN = re.compile(
    r"\[RecordingMicroAllocator\] Arena allocation (total|head|tail) (\d+) bytes"
)


def split_blocks(text: str) -> list[list[str]]:
    blocks = [block.strip().splitlines() for block in text.strip().split("\n\n") if block.strip()]
    if len(blocks) != 2:
        raise ValueError(f"Expected exactly 2 blocks (int8, float), found {len(blocks)}")
    return blocks


def parse_block(lines: list[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in lines:
        match = METRIC_PATTERN.fullmatch(line.strip())
        if not match:
            raise ValueError(f"Unrecognized line format: {line}")
        metric, value = match.groups()
        values[metric] = int(value)

    missing = {"total", "head", "tail"} - set(values)
    if missing:
        raise ValueError(f"Missing metrics in block: {sorted(missing)}")

    return values


def pct_change(new: int, old: int) -> float:
    if old == 0:
        return float("inf") if new != 0 else 0.0
    return (new - old) * 100.0 / old


def fmt_pct(value: float) -> str:
    if value == float("inf"):
        return "inf%"
    return f"{value:+.2f}%"


def print_comparison(int8_vals: dict[str, int], float_vals: dict[str, int]) -> None:
    print("Memory comparison (DRAM arena)")
    print("- int8 block: first block")
    print("- float block: second block")
    print("- head = non-persistent, tail = persistent")
    print()

    rows = [
        ("total", "Total DRAM"),
        ("head", "Head (non-persistent)"),
        ("tail", "Tail (persistent)"),
    ]

    header = f"{'Metric':<28} {'int8 (bytes)':>14} {'float (bytes)':>14} {'delta (float-int8)':>20} {'change':>10}"
    print(header)
    print("-" * len(header))

    for key, label in rows:
        int8_value = int8_vals[key]
        float_value = float_vals[key]
        delta = float_value - int8_value
        change = pct_change(float_value, int8_value)
        print(
            f"{label:<28} {int8_value:>14} {float_value:>14} {delta:>20} {fmt_pct(change):>10}"
        )


def plot_comparison(int8_vals: dict[str, int], float_vals: dict[str, int], output_path: Path) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping image generation.")
        return

    metric_keys = ["total", "head", "tail"]
    labels = ["Total DRAM", "Head (non-persistent)", "Tail (persistent)"]
    int8_data = [int8_vals[key] for key in metric_keys]
    float_data = [float_vals[key] for key in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_int8 = ax.bar(x - width / 2, int8_data, width, label="int8", color="#1f77b4")
    bars_float = ax.bar(x + width / 2, float_data, width, label="float", color="#ff7f0e")

    ax.set_ylabel("Bytes")
    ax.set_title("DRAM Memory Comparison (int8 vs float)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend()

    for bars in (bars_int8, bars_float):
        for bar in bars:
            height = int(bar.get_height())
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DRAM memory usage from RecordingMicroAllocator logs."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).with_name("memory_compare.txt"),
        help="Path to memory comparison text file (default: memory_compare.txt in this folder)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to output PNG (default: same name as input with .png)",
    )
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8")
    int8_block, float_block = split_blocks(text)

    int8_vals = parse_block(int8_block)
    float_vals = parse_block(float_block)

    print_comparison(int8_vals, float_vals)

    output_path = args.out if args.out is not None else args.file.with_suffix(".png")
    plot_comparison(int8_vals, float_vals, output_path)


if __name__ == "__main__":
    main()
