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
    def forward(self, body_emb, cloth_emb, body_vecs):

        B = body_emb.size(0)
        t = self.temperature

        # similarity logits
        logits = torch.matmul(body_emb, cloth_emb.T) / t

        # --------------------------------------------------------
        # 1. MASK FALSE NEGATIVES (CRITICAL)
        # --------------------------------------------------------
        fn_mask = self._false_negative_mask(body_vecs)

        logits_debiased = logits.masked_fill(fn_mask, float("-inf"))

        # --------------------------------------------------------
        # 2. HARD NEGATIVE EMPHASIS (SAFE VERSION)
        # --------------------------------------------------------
        if self.hard_neg_weight > 0:
            with torch.no_grad():
                eye = torch.eye(B, device=body_emb.device).bool()
                neg_mask = ~eye & ~fn_mask

                neg_logits = logits.masked_fill(~neg_mask, float("-inf"))

                hardness = torch.softmax(neg_logits, dim=1)

                boost = (self.hard_neg_weight * hardness).clamp(max=0.5)

            logits_debiased = logits_debiased + boost * neg_mask.float()

        # --------------------------------------------------------
        # 3. SAFE CROSS ENTROPY
        # --------------------------------------------------------
        labels = torch.arange(B, device=body_emb.device)

        # Guard: avoid all -inf rows
        has_finite = torch.isfinite(logits_debiased).any(dim=1)

        if not has_finite.all():
            logits_debiased = torch.where(
                has_finite.unsqueeze(1),
                logits_debiased,
                logits
            )

        loss_b2c = F.cross_entropy(logits_debiased, labels)
        loss_c2b = F.cross_entropy(logits_debiased.T, labels)

        loss = 0.5 * (loss_b2c + loss_c2b)

        # --------------------------------------------------------
        # FINAL SAFETY (NaN guard)
        # --------------------------------------------------------
        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=body_emb.device, requires_grad=True)

        return loss, t.item()


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