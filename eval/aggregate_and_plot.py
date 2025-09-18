# eval/aggregate_and_plot.py
import os, csv, numpy as np, matplotlib.pyplot as plt
from config.config import TrainConfig

def aggregate_by_bucket(csv_episode_log: str, bucket_size: int, out_csv: str):
    rows = []
    with open(csv_episode_log, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({
                "episode": int(r["episode"]),
                "reward": float(r["reward"]),
                "moves": int(r["moves"]),
                "checks": int(r["checks"]),
                "mates": int(r["mates"]),
            })
    if not rows: return

    max_ep = max(r["episode"] for r in rows)
    buckets = []
    for start in range(0, max_ep+1, bucket_size):
        end = min(start + bucket_size - 1, max_ep)
        sel = [r for r in rows if start <= r["episode"] <= end]
        if not sel: continue
        buckets.append({
            "range": f"{start}-{end}",
            "reward_mean": float(np.mean([r["reward"] for r in sel])),
            "moves_mean":  float(np.mean([r["moves"]  for r in sel])),
            "checks_mean": float(np.mean([r["checks"] for r in sel])),
            "mates_mean":  float(np.mean([r["mates"]  for r in sel])),
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["range","reward_mean","moves_mean","checks_mean","mates_mean"])
        for b in buckets:
            wr.writerow([b["range"], b["reward_mean"], b["moves_mean"], b["checks_mean"], b["mates_mean"]])
    return buckets

def plot_bucket_curves(buckets, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    labels = [b["range"] for b in buckets]
    x = list(range(len(labels)))

    def _bar(values, title, fname):
        plt.figure(figsize=(14,6))
        plt.bar(x, values)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print("[save]", path)

    _bar([b["reward_mean"] for b in buckets], "Average Reward per Bucket", "reward.png")
    _bar([b["moves_mean"]  for b in buckets], "Average Total Moves per Bucket", "moves.png")
    _bar([b["checks_mean"] for b in buckets], "Average Checks per Bucket", "checks.png")
    _bar([b["mates_mean"]  for b in buckets], "Average Checkmates per Bucket", "mates.png")

if __name__ == "__main__":
    cfg = TrainConfig()
    buckets = aggregate_by_bucket(cfg.csv_episode_log, cfg.bucket_size, cfg.csv_bucket_summary)
    if buckets:
        plot_bucket_curves(buckets, cfg.plots_dir)
