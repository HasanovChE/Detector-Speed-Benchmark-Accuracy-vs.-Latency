import argparse
import pandas as pd
from ultralytics import YOLO
from benchmark.common import (
    CONFIGS, RESULTS, DATA_YAML,
    ensure_results, choose_device, validate_weights,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ensure_results()
    validate_weights()
    device = choose_device(args.device)

    rows = []
    for cfg in CONFIGS:
        print(f"Evaluating {cfg['name']}")
        model = YOLO(str(cfg["weights"]))

        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=cfg["imgsz"],
            device=device,
            conf=0.001,
            iou=0.7,
            rect=False,
            plots=True,
            verbose=True,
        )

        box = metrics.box
        rows.append({
            "configuration": cfg["name"],
            "model": cfg["weights"].name,
            "imgsz": cfg["imgsz"],
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        })

        print(
            f"Precision={box.mp:.4f} | "
            f"Recall={box.mr:.4f} | "
            f"mAP50={box.map50:.4f} | "
            f"mAP50-95={box.map:.4f}"
        )

    output = RESULTS / "accuracy_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print("Saved:", output)

if __name__ == "__main__":
    main()