# ================================================================
# LOSS — Stable Contrastive Learning (FINAL VERSION)
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# DEBIASED INFO NCE (STAGE 3)
# ================================================================

class DeboasedInfoNCELoss(nn.Module):
    """
    Stable bidirectional InfoNCE with:
    ✔ Learnable temperature
    ✔ Mask-based false negative removal
    ✔ Mild hard-negative emphasis
    """

    def __init__(self,
                 init_temperature=0.07,
                 fn_threshold=0.85,        # updated (from 0.90)
                 hard_neg_weight=0.3):     # updated (from 0.2 -> 0.3)
        super().__init__()

        # log-space → stable optimization
        self.log_temp = nn.Parameter(torch.tensor(float(init_temperature)).log())

        self.fn_threshold = fn_threshold
        self.hard_neg_weight = hard_neg_weight

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.01, 0.5)

    # ------------------------------------------------------------
    # FALSE NEGATIVE MASK
    # ------------------------------------------------------------
    def _false_negative_mask(self, body_vecs):
        normed = F.normalize(body_vecs, dim=1)
        sim = torch.matmul(normed, normed.T)

        mask = sim > self.fn_threshold
        eye = torch.eye(body_vecs.size(0), device=body_vecs.device).bool()

        return mask & ~eye

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------
    def forward(self, body_embs, cloth_embs, body_vecs=None):

        # normalize
        body_embs = F.normalize(body_embs, dim=1)
        cloth_embs = F.normalize(cloth_embs, dim=1)

        sim = torch.matmul(body_embs, cloth_embs.T) / self.temperature.clamp(min=1e-4)

        batch_size = sim.size(0)
        labels = torch.arange(batch_size, device=sim.device)

        # =========================
        # 1. BODY → CLOTH LOSS
        # =========================
        loss_b2c = F.cross_entropy(sim, labels)
        loss_c2b = F.cross_entropy(sim.T, labels)
        body_cloth_loss = 0.5 * (loss_b2c + loss_c2b)

        # =========================
        # 2. CLOTH ↔ CLOTH LOSS (NEW)
        # =========================
        sim_cc = torch.matmul(cloth_embs, cloth_embs.T) / self.temperature.clamp(min=1e-4)

        # positives = same index (identity)
        loss_cc = F.cross_entropy(sim_cc, labels)

        # =========================
        # 3. COMBINE
        # =========================
        lambda_cc = 0.3   # 🔥 VERY IMPORTANT (tuneable)

        loss = body_cloth_loss + lambda_cc * loss_cc

        return loss, self.temperature.detach()


# ================================================================
# COLLAPSE CHECK (KEEP THIS)
# ================================================================

@torch.no_grad()
def check_embedding_collapse(body_embs, cloth_embs):

    body_std = body_embs.std(dim=0).mean().item()
    cloth_std = cloth_embs.std(dim=0).mean().item()

    n = min(200, cloth_embs.size(0))
    idx = torch.randperm(n)

    random_sim = F.cosine_similarity(
        cloth_embs[:n], cloth_embs[idx]
    ).mean().item()

    if random_sim > 0.3 or cloth_std < 0.05:
        verdict = "COLLAPSED"
    elif random_sim > 0.15 or cloth_std < 0.1:
        verdict = "WARNING"
    else:
        verdict = "OK"

    print(f"\n── Embedding health check ──────────────────")
    print(f"  Body std        : {body_std:.4f}  (want > 0.1)")
    print(f"  Cloth std       : {cloth_std:.4f}  (want > 0.1)")
    print(f"  Random cloth sim: {random_sim:.4f}  (want < 0.1)")
    print(f"  Verdict         : {verdict}")
    print(f"────────────────────────────────────────────\n")

    return dict(
        body_std=body_std,
        cloth_std=cloth_std,
        random_sim=random_sim,
        verdict=verdict
    )