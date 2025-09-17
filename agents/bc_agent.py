# agents/bc_agent.py
import numpy as np, torch, torch.nn.functional as F, chess
from expert.board_encoding import board_to_planes
from expert.action_encoder import ActionEncoder
from models.bc_policy_net import BCPolicyNet

class BCAgent:
    def __init__(self, model: BCPolicyNet, encoder: ActionEncoder, device="cuda", temperature=1.0):
        self.model = model.to(device).eval(); self.encoder = encoder
        self.device = device; self.temperature = max(1e-6, float(temperature))

    def _state_tensor(self, board: chess.Board):
        x = board_to_planes(board).astype(np.float32)/255.0  # (8,8,P)
        x = torch.from_numpy(np.transpose(x,(2,0,1))[None]).to(self.device)  # (1,P,8,8)
        return x

    def select_move(self, board: chess.Board, greedy=True) -> chess.Move:
        mask_np = self.encoder.valid_action_mask(board)
        if not mask_np.any(): return None
        x = self._state_tensor(board); mask = torch.from_numpy(mask_np)[None].to(self.device)
        with torch.no_grad():
            logits = self.model(x, mask=None)
            if greedy: aid = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = F.softmax(logits/self.temperature, dim=-1)[0]
                valid = torch.where(mask[0])[0].cpu().numpy()
                p = probs[valid].detach().cpu().numpy(); p = p / max(1e-9, p.sum())
                aid = int(np.random.choice(valid, p=p))
        return self.encoder.to_move(aid, board)
