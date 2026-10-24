from pathlib import Path
import cv2
from benchmark.common import load_dataset_config, ROOT

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    cfg = load_dataset_config()
    images_dir = cfg["_test_path"]
    labels_dir = ROOT / "dataset" / "labels" / "test"

    images = [
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in EXTENSIONS
    ]

    if not images:
        raise RuntimeError("No test images found.")

    problems = 0
    for image_path in images:
        if cv2.imread(str(image_path)) is None:
            print("Unreadable image:", image_path)
            problems += 1

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print("Missing label:", label_path)
            problems += 1
            continue

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            fields = line.split()
            try:
                if len(fields) != 5:
                    raise ValueError("Expected 5 fields")
                cls = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if cls != 0:
                    raise ValueError("Expected class ID 0")
                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    raise ValueError("Invalid normalized box")
            except ValueError as exc:
                print(
                    f"{label_path}:{line_number}: {exc}"
                )
                problems += 1

    print("Test images:", len(images))
    print("Dataset problems:", problems)

    if problems:
        raise SystemExit("Fix dataset issues before benchmarking.")
        
    print("Basic dataset checks passed.")

if __name__ == "__main__":
    main()