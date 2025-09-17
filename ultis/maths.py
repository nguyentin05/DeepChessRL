# utils/maths.py
import math, random, numpy as np, torch

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def wilson_ci(win: int, draw: int, loss: int, z: float = 1.96):
    n = max(1, win + draw + loss); s = win + 0.5 * draw; p = s / n
    denom = 1 + z*z/n; center = p + z*z/(2*n)
    rad = z * math.sqrt((p*(1-p)+z*z/(4*n))/n)
    return p, max(0.0,(center-rad)/denom), min(1.0,(center+rad)/denom)
