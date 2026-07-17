#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=11,<13"]
# ///
"""Render the committed Daniel LFM2 loss history as a blog-ready PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900
PLOT = (150, 150, 1510, 740)
BACKGROUND = "#0B0D0F"
GRID = "#30363A"
TEXT = "#F1F4F2"
MUTED = "#9CA6A1"
TRAIN = "#B7F24A"
VALIDATION = "#FF7C66"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("assets/data/daniel-lfm2-training-metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/images/daniel-lfm2-loss.png"),
    )
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def moving_average(values: list[float], radius: int = 2) -> list[float]:
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    train = metrics["train"]
    validation = metrics["validation"]
    train_smoothed = moving_average([point["loss"] for point in train])

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = PLOT
    x_max = 3.0
    y_max = 2.0

    def xy(epoch: float, loss: float) -> tuple[int, int]:
        x = left + int((epoch / x_max) * (right - left))
        y = bottom - int((loss / y_max) * (bottom - top))
        return x, y

    draw.text((left, 48), "Daniel LFM2-350M loss", fill=TEXT, font=font(44, bold=True))
    draw.text(
        (left, 102),
        "Completion-only cross-entropy · LoRA SFT · best checkpoint selected by validation loss",
        fill=MUTED,
        font=font(24),
    )

    for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
        _, y = xy(0, tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((65, y - 13), f"{tick:.1f}", fill=MUTED, font=font(22))
    for epoch in (0, 1, 2, 3):
        x, _ = xy(epoch, 0)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x - 8, bottom + 20), str(epoch), fill=MUTED, font=font(22))

    raw_points = [xy(point["epoch"], point["loss"]) for point in train]
    draw.line(raw_points, fill="#55722B", width=3)
    for x, y in raw_points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#709536")
    smooth_points = [xy(point["epoch"], loss) for point, loss in zip(train, train_smoothed)]
    draw.line(smooth_points, fill=TRAIN, width=8, joint="curve")

    validation_points = [xy(point["epoch"], point["loss"]) for point in validation]
    draw.line(validation_points, fill=VALIDATION, width=7)
    for x, y in validation_points:
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=VALIDATION, outline=BACKGROUND, width=4)

    best = min(validation, key=lambda point: point["loss"])
    best_x, best_y = xy(best["epoch"], best["loss"])
    draw.rounded_rectangle(
        (best_x + 24, best_y - 66, best_x + 310, best_y - 13),
        radius=6,
        fill="#171B1F",
        outline=VALIDATION,
        width=2,
    )
    draw.text(
        (best_x + 40, best_y - 55),
        f"best: {best['loss']:.3f} at epoch {best['epoch']:.0f}",
        fill=TEXT,
        font=font(22, bold=True),
    )

    draw.text((left, 790), "Epoch", fill=MUTED, font=font(22))
    draw.line((1000, 806, 1065, 806), fill=TRAIN, width=7)
    draw.text((1082, 790), "Train loss (5-point mean)", fill=TEXT, font=font(22))
    draw.line((1000, 850, 1065, 850), fill=VALIDATION, width=7)
    draw.text((1082, 834), "Validation loss", fill=TEXT, font=font(22))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, optimize=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
