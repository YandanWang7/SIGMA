import argparse
import json
import math
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser("Build FRACAL calibration stats from LVIS annotations")
    parser.add_argument(
        "--ann",
        default="data/annotations/lvis_v1_train.json",
        help="Path to lvis_v1_train.json.",
    )
    parser.add_argument(
        "--out",
        default="config/calibration/lvis_fracal_stats.pth",
        help="Output .pth file.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1204,
        help="Classifier output dimension. LT-DETR-SIGMA LVIS uses 1204 with class 0 unused.",
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
        help="Grid sizes used for box-counting fractal dimension.",
    )
    return parser.parse_args()


def fit_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 1.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return 1.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def main():
    args = parse_args()
    ann_path = Path(args.ann)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with ann_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_sizes = {
        img["id"]: (float(img["width"]), float(img["height"]))
        for img in data.get("images", [])
    }

    counts = torch.zeros(args.num_classes, dtype=torch.float32)
    centers_by_class = [[] for _ in range(args.num_classes)]

    for ann in data.get("annotations", []):
        cat_id = int(ann["category_id"])
        if cat_id < 0 or cat_id >= args.num_classes:
            continue
        width, height = image_sizes.get(ann["image_id"], (None, None))
        if not width or not height:
            continue
        x, y, w, h = ann["bbox"]
        cx = min(max((float(x) + 0.5 * float(w)) / width, 0.0), 1.0 - 1e-7)
        cy = min(max((float(y) + 0.5 * float(h)) / height, 0.0), 1.0 - 1e-7)
        counts[cat_id] += 1
        centers_by_class[cat_id].append((cx, cy))

    prior = counts / counts.sum().clamp_min(1.0)
    phi = torch.ones(args.num_classes, dtype=torch.float32)

    log_grids = [math.log(g) for g in args.grid_sizes]
    for cat_id, centers in enumerate(centers_by_class):
        if len(centers) < 2:
            continue
        occupied_counts = []
        for grid in args.grid_sizes:
            occupied = set()
            for cx, cy in centers:
                gx = min(int(cx * grid), grid - 1)
                gy = min(int(cy * grid), grid - 1)
                occupied.add((gx, gy))
            occupied_counts.append(max(len(occupied), 1))
        log_occupied = [math.log(v) for v in occupied_counts]
        phi[cat_id] = float(max(1e-6, min(2.0, fit_slope(log_grids, log_occupied))))

    stats = {
        "class_counts": counts,
        "fracal_prior": prior,
        "fracal_phi": phi,
        "meta": {
            "ann": str(ann_path),
            "num_classes": args.num_classes,
            "grid_sizes": args.grid_sizes,
            "class_0": "unused neutral slot for LT-DETR-SIGMA LVIS labels",
        },
    }
    torch.save(stats, out_path)
    nonzero = int((counts > 0).sum().item())
    print(f"saved {out_path}")
    print(f"classes with instances: {nonzero}/{args.num_classes}")
    print(f"total instances: {int(counts.sum().item())}")


if __name__ == "__main__":
    main()
