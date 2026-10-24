# Detector Speed Benchmark: Accuracy vs. Latency

This project benchmarks YOLO-based license plate detector configurations, comparing their accuracy and inference speed on a consistent dataset and hardware. The primary goal is to identify a suitable configuration for a real-time parking-gate system, targeting under 100 ms latency per frame.

## Features

- Benchmark multiple YOLO detector configurations.
- Utilize a consistent test dataset.
- Measure inference latency with repeated runs using `time.perf_counter()`.
- Evaluate Precision, Recall, mAP50, and mAP50-95.
- Calculate FPS, mean, median, and P95 latency.
- Generate an "Accuracy vs. Latency" chart.
- Apply a 100 ms real-time constraint for recommendations.
- Provide configuration recommendations based on measured results.
- Optional comparisons: CPU/GPU, input-resolution, 1000-image throughput.

## Default Configurations

The benchmark compares the following YOLOv8 configurations:

| Configuration | Model   | Input Size |
| :------------ | :------ | :--------- |
| YOLOv8n_640   | YOLOv8n | 640x640    |
| YOLOv8s_640   | YOLOv8s | 640x640    |
| YOLOv8n_416   | YOLOv8n | 416x416    |

_The YOLOv8n_416 configuration demonstrates the effect of reduced input resolution._

## Project Structure

```
detector-speed-benchmark/
│
├── models/
│   ├── yolov8n_best.pt
│   └── yolov8s_best.pt
│
├── dataset/
│   ├── images/
│   │   └── test/
│   ├── labels/
│   │   └── test/
│   └── data.yaml
│
├── benchmark/
│   ├── benchmark.py
│   ├── evaluate.py
│   └── plot_results.py
│
├── results/
│   ├── accuracy_results.csv
│   ├── benchmark_results.csv
│   ├── final_results.csv
│   └── accuracy_vs_latency.png
│
├── requirements.txt
└── README.md
```

- `benchmark/common.py`: Defines shared configurations, directory paths, device selection, and hardware information extraction.
- `benchmark/check_dataset.py`: Validates the integrity of test images and YOLO label formats prior to evaluation.
- `benchmark/evaluate.py`: Calculates detection accuracy metrics (Precision, Recall, mAP50, mAP50-95) using the Ultralytics validation engine.
- `benchmark/benchmark.py`: The core timing script that measures inference latency (mean, median, P95, FPS) using multiple runs and warm-up cycles.
- `benchmark/plot_results.py`: Merges speed and accuracy data, evaluates strict real-time constraints (<100ms), and generates the final Accuracy vs. Latency chart.

## Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
2.  **Activate environment (Windows):**
    ```bash
    venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The benchmark uses a consistent license-plate test dataset (`dataset/data.yaml`) for all configurations to ensure reliable comparisons.

Example `data.yaml`:

```yaml
path: dataset

train: images/train
val: images/val
test: images/test

names:
  0: license_plate
```

## Usage

### 1. Accuracy Evaluation

Run:

```bash
python benchmark/evaluate.py
```

Records Precision, Recall, mAP50, and mAP50-95. Results are saved to `results/accuracy_results.csv`.

### 2. Speed Benchmark

Run:

```bash
python benchmark/benchmark.py
```

Performs warm-up inferences and repeated tests for each configuration. Records mean, median, standard deviation, min/max, P95 latency, and FPS. Results are saved to `results/benchmark_results.csv`.

### 3. Generate Accuracy vs. Latency Chart

After both benchmarks, run:

```bash
python benchmark/plot_results.py
```

This combines results into `results/final_results.csv` and generates `results/accuracy_vs_latency.png`, visualizing the trade-off between detection accuracy and inference latency.

## Real-Time Requirement

The target parking-gate system requires a maximum detector latency of **100 ms/frame** (approx. 10 FPS). Recommendations prioritize configurations with a mean latency below this threshold. A stricter evaluation may also require P95 latency < 100 ms.

## Recommendation

The final configuration should be selected based on measured benchmark results, prioritizing those that meet the 100 ms latency requirement and provide the necessary accuracy for the application. Theoretical model speeds should not be used as a basis.

## Hardware Information

Benchmark results are hardware-dependent. Please record the following:

- **CPU:**
- **RAM:**
- **GPU:**
- **Operating System:**
- **Python:**
- **PyTorch:**
- **Ultralytics:**
- **CUDA:**

## Important Benchmark Notes

To ensure a fair comparison:

- Consistent test dataset, hardware, preprocessing, and confidence settings across all models.
- Multiple inference runs with warm-up runs.
- GPU synchronization when CUDA is available.
- Inference time measured using `Python time.perf_counter()`.

## Limitations

This benchmark focuses solely on **detector inference latency**. It does **not** include:

- OCR latency
- Image capture latency
- Network communication
- Country/format classification
- Database operations
- Gate actuator response time

Therefore, the total system latency will be higher than the reported detector-only latency.

## Technologies Used

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Expected Output

Upon completion, the project will generate:

- Accuracy results
- Latency results
- Accuracy vs. Latency chart
- Hardware information
- 100 ms constraint analysis
- Final configuration recommendation

## Reference

- [Ultralytics Benchmark Documentation](https://docs.ultralytics.com/modes/benchmark/)

## Benchmark Results

**Hardware Environment**

- **OS:** Windows 10
- **CPU:** Intel64 Family 6 Model 142 Stepping 10 (8 Logical Cores)
- **Device:** CPU (No CUDA)
- **Frameworks:** Python 3.12.0, PyTorch 2.14.1+cpu, Ultralytics 8.4.187

**Accuracy & Latency Metrics**
| Configuration | Image Size | mAP50 | mAP50-95 | Mean Latency (ms) | P95 Latency (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv8_416 | 416x416 | 0.995 | 0.7532 | 144.97 | 327.54 | 6.90 |
| YOLOv8_640 | 640x640 | 0.995 | 0.7495 | 233.70 | 537.18 | 4.28 |

**Accuracy vs. Latency**
![Accuracy vs Latency](accuracy_vs_latency.png)

**Conclusion & Recommendation**
Based on the benchmark testing, **no configuration meets both the mean and P95 latency requirements below 100 ms** on the current CPU hardware.

While `YOLOv8_416` achieved a slightly higher mAP50-95 (0.7532) and significantly faster mean latency (144.97 ms) than the 640 resolution, it still falls short of the strict 10 FPS / <100ms real-time constraint.

**Next Steps for Deployment:**
To meet the real-time requirements for a parking-gate system, hardware or optimization upgrades are necessary. Consider exporting the model to ONNX or OpenVINO for CPU acceleration, or deploying the application on a dedicated GPU edge device.
