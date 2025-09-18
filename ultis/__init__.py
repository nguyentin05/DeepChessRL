# utils/__init__.py
import numpy as np, torch as T, os, cv2

def make_batch_ids(n: int, batch_size: int, shuffle: bool = True):
    starts = np.arange(0, n, batch_size)
    indices = np.arange(n, dtype=np.int64)
    if shuffle: np.random.shuffle(indices)
    return [indices[i:i+batch_size] for i in starts]

def save_to_video(path: str, frames: np.ndarray, fps: int = 2):
    if len(frames) == 0: return
    size = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.VideoWriter(path, fourcc, fps, size)
    for f in frames: out.write(f)
    out.release()

def set_cpu_threads(n: int = 4):
    T.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
