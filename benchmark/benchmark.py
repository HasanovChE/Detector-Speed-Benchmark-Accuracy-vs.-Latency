import argparse
import time
import json
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from benchmark.common import (
    CONFIGS, RESULTS,
    ensure_results, choose_device, validate_weights,
    load_dataset_config, hardware_info,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def synchronize(device):
    if device.isdigit() or device.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0:
        parser.error("runs must be >= 1 and warmup must be >= 0")

    ensure_results()
    validate_weights()
    cfg_data = load_dataset_config()
    device = choose_device(args.device)

    image_paths = sorted(
        p for p in cfg_data["_test_path"].rglob("*")
        if p.suffix.lower() in EXTENSIONS
    )

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        raise RuntimeError("No test images found.")

    # Decode images before timing to exclude disk I/O.
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append((path, image))

    if not images:
        raise RuntimeError("No readable test images.")

    rows = []
    for cfg in CONFIGS:
        print(f"\nBenchmarking {cfg['name']}")
        model = YOLO(str(cfg["weights"]))

        # Warm-up runs are excluded from statistics.
        for _ in range(args.warmup):
            model.predict(
                source=images[0][1],
                imgsz=cfg["imgsz"],
                device=device,
                conf=args.conf,
                verbose=False,
                save=False,
                rect=False,
            )

        synchronize(device)
        latencies = []

        for run_index in range(args.runs):
            for _, image in images:
                synchronize(device)
                start = time.perf_counter()
                
                model.predict(
                    source=image,
                    imgsz=cfg["imgsz"],
                    device=device,
                    conf=args.conf,
                    verbose=False,
                    save=False,
                    rect=False,
                )
                
                synchronize(device)
                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000.0
                latencies.append(elapsed_ms)
                
            print(f"Completed pass {run_index + 1}/{args.runs}")

        values = np.asarray(latencies, dtype=float)
        mean_ms = float(np.mean(values))

        rows.append({
            "configuration": cfg["name"],
            "model": cfg["weights"].name,
            "imgsz": cfg["imgsz"],
            "device": device,
            "images_per_pass": len(images),
            "timed_inferences": len(values),
            "warmup_runs": args.warmup,
            "repeat_passes": args.runs,
            "mean_latency_ms": mean_ms,
            "median_latency_ms": float(np.median(values)),
            "std_latency_ms": float(np.std(values, ddof=1))
            if len(values) > 1 else 0.0,
            "min_latency_ms": float(np.min(values)),
            "max_latency_ms": float(np.max(values)),
            "p95_latency_ms": float(np.percentile(values, 95)),
            "fps_from_mean_latency": 1000.0 / mean_ms,
        })

        print(
            f"Mean: {mean_ms:.2f} ms | "
            f"Median: {np.median(values):.2f} ms | "
            f"P95: {np.percentile(values, 95):.2f} ms | "
            f"FPS: {1000 / mean_ms:.2f}"
        )

    pd.DataFrame(rows).to_csv(
        RESULTS / "benchmark_results.csv", index=False
    )
    (RESULTS / "hardware_info.json").write_text(
        json.dumps(hardware_info(device), indent=2),
        encoding="utf-8",
    )

    print("\nSaved benchmark_results.csv")
    print("Saved hardware_info.json")

if __name__ == "__main__":
    main()