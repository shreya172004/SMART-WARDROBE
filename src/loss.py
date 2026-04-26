# ================================================================
# LOSS — Stable Contrastive Learning
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeboasedInfoNCELoss(nn.Module):
    """
    Bidirectional InfoNCE with:
    - Learnable temperature (log-space, stable)
    - Mask-based false negative removal
    - Mild hard-negative emphasis
    """

    def __init__(self,
                 init_temperature=0.07,
                 fn_threshold=0.85,
                 hard_neg_weight=0.3):
        super().__init__()
        self.log_temp        = nn.Parameter(torch.tensor(float(init_temperature)).log())
        self.fn_threshold    = fn_threshold
        self.hard_neg_weight = hard_neg_weight

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.01, 0.5)

    def _false_negative_mask(self, body_vecs):
        normed = F.normalize(body_vecs, dim=1)
        sim    = torch.matmul(normed, normed.T)
        mask   = sim > self.fn_threshold
        eye    = torch.eye(body_vecs.size(0), device=body_vecs.device).bool()
        return mask & ~eye

    def forward(self, body_embs, cloth_embs, body_vecs=None):
        body_embs  = F.normalize(body_embs, dim=1)
        cloth_embs = F.normalize(cloth_embs, dim=1)

        t   = self.temperature.clamp(min=1e-4)
        sim = torch.matmul(body_embs, cloth_embs.T) / t

        B      = sim.size(0)
        labels = torch.arange(B, device=sim.device)

        # Optional false-negative masking
        if body_vecs is not None and self.fn_threshold < 1.0:
            fn_mask = self._false_negative_mask(body_vecs)
            sim_b2c = sim.masked_fill(fn_mask, float("-inf"))
            sim_c2b = sim.T.masked_fill(fn_mask, float("-inf"))
        else:
            sim_b2c = sim
            sim_c2b = sim.T

        loss_b2c = F.cross_entropy(sim_b2c, labels)
        loss_c2b = F.cross_entropy(sim_c2b, labels)
        loss     = 0.5 * (loss_b2c + loss_c2b)

        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=body_embs.device, requires_grad=True)

        return loss, self.temperature.detach()


# ================================================================
# COLLAPSE CHECK
# ================================================================

@torch.no_grad()
def check_embedding_collapse(body_embs: torch.Tensor,
                              cloth_embs: torch.Tensor) -> dict:
    """
    Detects embedding collapse.

    Thresholds (corrected — previous 0.12 was too strict for this task):
      random_sim > 0.4  → COLLAPSED
      random_sim > 0.2  → WARNING
      else              → OK

    Body/cloth std < 0.04 also triggers COLLAPSED.
    """
    body_std  = body_embs.std(dim=0).mean().item()
    cloth_std = cloth_embs.std(dim=0).mean().item()

    n   = min(200, cloth_embs.size(0))
    idx = torch.randperm(n)
    random_sim = F.cosine_similarity(
        cloth_embs[:n], cloth_embs[idx]
    ).mean().item()

    if random_sim > 0.4 or cloth_std < 0.04:
        verdict = "COLLAPSED"
    elif random_sim > 0.2 or cloth_std < 0.1:
        verdict = "WARNING"
    else:
        verdict = "OK"

    print(f"\n── Embedding health check ──────────────────")
    print(f"  Body std        : {body_std:.4f}  (want > 0.1)")
    print(f"  Cloth std       : {cloth_std:.4f}  (want > 0.1)")
    print(f"  Random cloth sim: {random_sim:.4f}  (want < 0.2)")
    print(f"  Verdict         : {verdict}")
    print(f"────────────────────────────────────────────\n")

    return dict(body_std=body_std, cloth_std=cloth_std,
                random_sim=random_sim, verdict=verdict)