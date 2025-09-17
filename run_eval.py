# run_eval.py
from config.config import EvalConfig
from eval.evaluator import run_progress_eval
from eval.plot_progress import plot_progress

if __name__ == "__main__":
    # Điền danh sách checkpoint của bạn ở đây
    cfg = EvalConfig(
        checkpoints=[
            "ckpts/bc_1k.pt",
            "ckpts/bc_5k.pt",
            "ckpts/bc_10k.pt",
            "ckpts/bc_full.pt",
            "ckpts/dagger_final.pt",
        ],
        device="cuda",  # hoặc "cpu"
        games_per_checkpoint=200,
        out_csv="results/progress_vs_heuristic.csv",
        out_png="results/progress_vs_heuristic.png",
    )
    run_progress_eval(cfg)
    plot_progress(cfg.out_csv, cfg.out_png)
