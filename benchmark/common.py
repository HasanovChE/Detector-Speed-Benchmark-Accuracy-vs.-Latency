from pathlib import Path
import platform
import sys
import torch
import ultralytics
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_YAML = ROOT / "dataset" / "data.yaml"

CONFIGS = [
    {
        "name": "YOLOv8_640",
        "weights": ROOT / "models" / "best.pt",
        "imgsz": 640,
    },
    {
        "name": "YOLOv8_416",
        "weights": ROOT / "models" / "best.pt",
        "imgsz": 416,
    },
]

def ensure_results():
    RESULTS.mkdir(parents=True, exist_ok=True)

def load_dataset_config():
    with open(DATA_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    
    test_path = Path(cfg["test"])
    if not test_path.is_absolute():
        test_path = base / test_path
        
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )
    
    cfg["_test_path"] = test_path.resolve()
    return cfg

def choose_device(requested="auto"):
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in PyTorch.")
    return requested

def validate_weights():
    missing = [
        str(c["weights"])
        for c in CONFIGS
        if not c["weights"].exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing trained weights:\n" + "\n".join(missing)
        )

def hardware_info(device):
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "device": device,
    }