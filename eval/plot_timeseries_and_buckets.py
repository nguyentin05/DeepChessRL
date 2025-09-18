# eval/plot_timeseries_and_buckets.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RESULTS_DIR = "results"
CSV_EPISODE_LOG = os.path.join(RESULTS_DIR, "episodes_log.csv")
PLOTS_PATH = os.path.join(RESULTS_DIR, "plots_timeseries.png")
PLOTS_BUCKETS_PATH = os.path.join(RESULTS_DIR, "plots_buckets.png")

# Tham số “chia nhỏ giai đoạn”
BUCKET_SIZE = 500        # 0-500, 500-1000, ...
MA_WIN = 32              # moving-average window cho time-series
MATE_DENS_WIN = 200      # cửa sổ tính mật độ checkmates
ALPHA = 0.25

# Màu như ví dụ bạn gửi
C_BLACK = "tab:blue"
C_WHITE = "tab:orange"


def moving_average(arr, count):
    if count <= 0 or len(arr) < count:
        return np.array([])
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    # MA[i] = mean(arr[i-count:i])
    ma = (cumsum[count:] - cumsum[:-count]) / float(count)
    return ma


def _load_new_csv(csv_path):
    """
    Kiểu mới: episodes_log.csv (agent 1 chiều)
    columns: episode,reward,moves,checks,mates
    """
    if not os.path.exists(csv_path):
        return None
    episodes, rewards, moves, checks, mates = [], [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            episodes.append(int(r["episode"]))
            rewards.append(float(r["reward"]))
            moves.append(int(r["moves"]))
            checks.append(int(r["checks"]))
            mates.append(int(r["mates"]))
    if not episodes:
        return None
    # agent 1 chiều -> convert thành (1, E) để tái sử dụng code plot
    E = max(episodes) + 1
    arr_reward = np.zeros((1, E), dtype=np.float32)
    arr_moves  = np.zeros((1, E), dtype=np.float32)
    arr_checks = np.zeros((1, E), dtype=np.float32)
    arr_mates  = np.zeros((1, E), dtype=np.float32)
    for ep, r, m, c, mt in zip(episodes, rewards, moves, checks, mates):
        arr_reward[0, ep] = r
        arr_moves[0, ep]  = m
        arr_checks[0, ep] = c
        arr_mates[0, ep]  = mt
    return arr_reward, arr_moves, arr_checks, arr_mates


def _load_legacy_npy(folder):
    """
    Kiểu cũ: npy (2, E) cho Black/White
    """
    try:
        moves = np.load(os.path.join(folder, "moves.npy"))
        mates = np.load(os.path.join(folder, "mates_win.npy"))
        checks = np.load(os.path.join(folder, "checks_win.npy"))
        rewards = np.load(os.path.join(folder, "rewards.npy"))
        return rewards, moves, checks, mates
    except Exception:
        return None


def plot_time_series(ax, arr2d, title, color_pair=(C_BLACK, C_WHITE), alpha=ALPHA, ma_win=MA_WIN):
    """
    arr2d shape:
      - (1, E) cho agent single
      - (2, E) cho Black/White
    """
    ax.set_title(title)
    E = arr2d.shape[1]
    ax.set_xlim([0, E])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")

    n_rows = arr2d.shape[0]
    for i in range(n_rows):
        color = color_pair[0] if i == 0 else color_pair[1]
        label = "Agent" if n_rows == 1 else ("Black" if i == 0 else "White")
        ax.plot(arr2d[i], alpha=alpha, c=color, label=label)
        ma = moving_average(arr2d[i], ma_win)
        if ma.size > 0:
            ax.plot(range(ma_win, ma_win + len(ma)), ma, c=color, alpha=1.0)
    ax.legend()
    ax.grid(True)


def plot_total_moves(ax, moves2d, title="Total Moves", alpha=ALPHA, ma_win=MA_WIN):
    # tổng 2 bên (nếu có 2 bên), hoặc agent nếu 1 bên
    if moves2d.shape[0] == 1:
        arr = moves2d[0]
    else:
        arr = moves2d.sum(axis=0)
    E = len(arr)
    ax.set_title(title)
    ax.set_xlim([0, E])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Moves")
    ax.plot(arr, alpha=alpha, c=C_BLACK)
    ma = moving_average(arr, ma_win)
    if ma.size > 0:
        ax.plot(range(ma_win, ma_win + len(ma)), ma, alpha=1.0, c=C_BLACK)
    ax.grid(True)


def density_checkmates(arr2d, count, episodes):
    """
    arr2d mates: (1,E) hoặc (2,E)
    Trả về mật độ checkmates theo cửa sổ 'count' (dùng max theo chiều người chơi như code mẫu).
    """
    a = arr2d.max(axis=0)  # nếu có 2 bên -> lấy max per-ep
    dens = []
    for i in range(episodes):
        j = max(0, i - count)
        dens.append(np.sum(a[j:i]) / max(1, (i - j)))
    return np.array(dens)


def bar_buckets(ax, arr2d, title, bucket_size):
    """
    Chia nhỏ giai đoạn thành nhiều cột (bucket bar chart).
    Với 1 chiều: hiển thị mean per bucket (Agent).
    Với 2 chiều: vẽ 2 nhóm cột cạnh nhau (Black/White).
    """
    E = arr2d.shape[1]
    ax.set_title(title)
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Mean Value")
    # xác định buckets
    buckets = []
    for start in range(0, E, bucket_size):
        end = min(start + bucket_size, E)
        buckets.append((start, end))
    labels = [f"{s}-{e-1}" for (s, e) in buckets]
    x = np.arange(len(buckets))

    if arr2d.shape[0] == 1:
        vals = [float(np.mean(arr2d[0, s:e])) for (s, e) in buckets]
        ax.bar(x, vals, color=C_BLACK, alpha=0.9, label="Agent")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis="y")
    else:
        vals_b = [float(np.mean(arr2d[0, s:e])) for (s, e) in buckets]
        vals_w = [float(np.mean(arr2d[1, s:e])) for (s, e) in buckets]
        width = 0.42
        ax.bar(x - width/2, vals_b, width, color=C_BLACK, alpha=0.9, label="Black")
        ax.bar(x + width/2, vals_w, width, color=C_WHITE, alpha=0.9, label="White")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis="y")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data_new = _load_new_csv(CSV_EPISODE_LOG)
    data_legacy = _load_legacy_npy(RESULTS_DIR)

    if data_new is None and data_legacy is None:
        print("Không tìm thấy dữ liệu. Hãy chạy training trước hoặc đặt .npy vào 'results/'.")
        return

    if data_new is not None:
        rewards, moves, checks, mates = data_new  # (1,E) each
    else:
        rewards, moves, checks, mates = data_legacy  # (2,E) each

    E = rewards.shape[1]

    # --- Figure 1: Time-series + MA + Mate density ---
    fig, axs = plt.subplots(2, 2, figsize=(20, 12), dpi=160)
    fig.suptitle(f"Time-series (Episodes = {E})")

    # Rewards
    plot_time_series(axs[0, 0], rewards, "Rewards", alpha=ALPHA, ma_win=MA_WIN)
    # Moves (total)
    plot_total_moves(axs[0, 1], moves, "Total Moves", alpha=ALPHA, ma_win=MA_WIN)
    # Checks
    plot_time_series(axs[1, 0], checks, "Checks", alpha=ALPHA, ma_win=MA_WIN)
    # Checkmates + density
    ax = axs[1, 1]
    ax2 = ax.twinx()
    dens = density_checkmates(mates, MATE_DENS_WIN, E)
    ax2.plot(range(E), dens, color="tab:green", alpha=1.0,
             label=f"Checkmate density (win={MATE_DENS_WIN})", linewidth=2)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper right")
    ax2.grid(False)
    plot_time_series(ax, mates, "Checkmates", alpha=ALPHA, ma_win=MA_WIN)
    fig.tight_layout()
    fig.savefig(PLOTS_PATH)
    plt.close(fig)
    print("[save]", PLOTS_PATH)

    # --- Figure 2: Bucket bars (chia nhỏ giai đoạn thành nhiều cột) ---
    fig2, axs2 = plt.subplots(2, 2, figsize=(22, 12), dpi=160)
    fig2.suptitle(f"Bucketed Means (Bucket size = {BUCKET_SIZE})")

    bar_buckets(axs2[0, 0], rewards, "Rewards (mean per bucket)", BUCKET_SIZE)
    bar_buckets(axs2[0, 1], moves,   "Moves (mean per bucket)",   BUCKET_SIZE)
    bar_buckets(axs2[1, 0], checks,  "Checks (mean per bucket)",  BUCKET_SIZE)
    bar_buckets(axs2[1, 1], mates,   "Checkmates (mean per bucket)", BUCKET_SIZE)

    fig2.tight_layout()
    fig2.savefig(PLOTS_BUCKETS_PATH)
    plt.close(fig2)
    print("[save]", PLOTS_BUCKETS_PATH)


if __name__ == "__main__":
    main()
