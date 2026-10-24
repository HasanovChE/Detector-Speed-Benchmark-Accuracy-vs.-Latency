import pandas as pd
import matplotlib.pyplot as plt
from benchmark.common import RESULTS, ensure_results

LIMIT_MS = 100.0

def main():
    ensure_results()
    accuracy_path = RESULTS / "accuracy_results.csv"
    speed_path = RESULTS / "benchmark_results.csv"

    if not accuracy_path.exists() or not speed_path.exists():
        raise FileNotFoundError(
            "Run evaluate and benchmark first."
        )

    accuracy = pd.read_csv(accuracy_path)
    speed = pd.read_csv(speed_path)

    final = accuracy.merge(
        speed,
        on=["configuration", "model", "imgsz"],
        how="inner",
        validate="one_to_one",
    )

    if final.empty:
        raise RuntimeError("No matching benchmark configurations.")

    final["mean_pass"] = final["mean_latency_ms"] < LIMIT_MS
    final["p95_pass"] = final["p95_latency_ms"] < LIMIT_MS
    final["realtime_pass"] = (
        final["mean_pass"] & final["p95_pass"]
    )

    final["estimated_1000_images_seconds"] = (
        final["mean_latency_ms"]
    )

    final["recommendation"] = ""
    eligible = final[final["realtime_pass"]]

    if not eligible.empty:
        best_index = eligible["map50_95"].idxmax()
        final.loc[best_index, "recommendation"] = "RECOMMENDED"
        recommended = final.loc[best_index, "configuration"]
        reason = (
            f"Recommended {recommended}: highest mAP50-95\n"
            "among configurations passing both latency limits."
        )
    else:
        reason = (
            "No configuration meets both mean and P95 latency\n"
            "requirements below 100 ms. Optimize or test faster\n"
            "hardware before deployment."
        )

    final.to_csv(RESULTS / "final_results.csv", index=False)

    # Accuracy versus mean detector latency.
    plt.figure(figsize=(10, 6))
    plt.scatter(
        final["mean_latency_ms"],
        final["map50_95"],
        s=100,
    )

    for _, row in final.iterrows():
        plt.annotate(
            row["configuration"],
            (row["mean_latency_ms"], row["map50_95"]),
            xytext=(6, 5),
            textcoords="offset points",
        )

    plt.axvline(
        LIMIT_MS,
        linestyle="--",
        label="100 ms latency limit",
    )

    plt.xlabel("Mean detector latency (ms/image)")
    plt.ylabel("mAP50-95")
    plt.title("License Plate Detection: Accuracy vs. Latency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS / "accuracy_vs_latency.png",
        dpi=180,
    )
    plt.close()

    (RESULTS / "recommendation.txt").write_text(
        reason
        + "\n\nStrict requirement: mean < 100 ms\n"
        + "AND P95 < 100 ms.\n"
        + "Detector-only latency; downstream system latency\n"
        + "will be higher.\n",
        encoding="utf-8",
    )

    columns = [
        "configuration",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "mean_latency_ms",
        "p95_latency_ms",
        "fps_from_mean_latency",
        "realtime_pass",
        "recommendation",
    ]

    print(final[columns].to_string(index=False))
    print("\n", reason)
    print("Reports saved in:", RESULTS)

if __name__ == "__main__":
    main()