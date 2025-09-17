# run_pipeline.py
from config.config import EvalConfig
from eval.evaluator import run_eval
from eval.plot_progress import plot_bars

if __name__=="__main__":
    cfg=EvalConfig(
        ckpt_dir="ckpts",
        games_per_checkpoint=200,
        device="cuda",  # hoặc "cpu"
        out_csv="results/progress_vs_heuristic.csv",
        out_png="results/progress_vs_heuristic.png",
    )
    run_eval(cfg)
    plot_bars(cfg.out_csv, cfg.out_png)
