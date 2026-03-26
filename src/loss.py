"""
loss.py — Stable contrastive losses for FashInclusioNet
 
Three problems fixed vs the original train.py final_loss():
  1. Temperature was 0.03  →  gradients explode / loss collapses to 0.
     Fixed: 0.07 (standard SimCLR value), with learnable parameter.
  2. Debiasing used gradient *scaling* (sim * debias)
     →  distorts the softmax denominator unpredictably.
     Fixed: MASK-based debiasing — false negatives are zeroed out
     from the denominator entirely (safe, no gradient distortion).
  3. Hard-negative weighting multiplied AFTER softmax
     →  double-normalizes and creates instability.
     Fixed: additive logit boost *before* the cross-entropy, clamped.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ================================================================
# LEARNABLE-TEMPERATURE InfoNCE  (Stage 1 + 2: pretraining)
# ================================================================
 
class InfoNCELoss(nn.Module):
    """
    Bidirectional InfoNCE with learnable temperature.
    Use this for DeepFashion and Polyvore pretraining where
    we don't need body-similarity debiasing.
    """
 
    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        # log-space for numerical stability; exp keeps it positive
        self.log_temp = nn.Parameter(torch.tensor(float(init_temperature)).log())
 
    @property
    def temperature(self):
        # Clamp to (0.01, 0.5) — prevents explosion in either direction
        return self.log_temp.exp().clamp(0.01, 0.5)
 
    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        emb_a, emb_b : (B, D) L2-normalized embeddings.
        Diagonal entries are positive pairs, off-diagonal are negatives.
        """
        B      = emb_a.size(0)
        t      = self.temperature
        logits = torch.matmul(emb_a, emb_b.T) / t          # (B, B)
        labels = torch.arange(B, device=emb_a.device)
 
        loss_ab = F.cross_entropy(logits,   labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2
 
 
# ================================================================
# DEBIASED InfoNCE  (Stage 3: Fashionista body-alignment)
# ================================================================
 
class DeboasedInfoNCELoss(nn.Module):
    """
    Bidirectional InfoNCE with:
      - Learnable temperature
      - Mask-based false-negative debiasing  (body-similarity aware)
      - Optional semi-hard negative emphasis
 
    This replaces the original final_loss() in train.py.
    """
 
    def __init__(self,
                 init_temperature: float = 0.07,
                 fn_threshold: float = 0.85,
                 hard_neg_weight: float = 0.3):
        """
        fn_threshold     : cosine similarity above which two body vectors
                           are treated as referring to the same body type
                           → their clothing pair is a likely false negative.
        hard_neg_weight  : how much to boost semi-hard negatives in logit
                           space. 0 = disabled, 0.5 = moderate.
        """
        super().__init__()
        self.log_temp       = nn.Parameter(torch.tensor(float(init_temperature)).log())
        self.fn_threshold   = fn_threshold
        self.hard_neg_weight = hard_neg_weight
 
    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.01, 0.5)
 
    def _false_negative_mask(self, body_vecs: torch.Tensor) -> torch.Tensor:
        """
        Returns boolean (B, B) mask where True = likely false negative.
        Diagonal is always False (never mask the actual positive pair).
 
        SAFE approach: we ZERO OUT these positions in the denominator
        rather than scaling gradients (which caused instability in v1).
        """
        normed  = F.normalize(body_vecs, dim=1)
        sim     = torch.matmul(normed, normed.T)          # (B, B)
        mask    = sim > self.fn_threshold
        eye     = torch.eye(body_vecs.size(0), dtype=torch.bool, device=body_vecs.device)
        return mask & ~eye
 
    def forward(self,
                body_emb:    torch.Tensor,
                cloth_emb:   torch.Tensor,
                body_vecs:   torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        body_emb    : (B, D) L2-normalized body embeddings
        cloth_emb   : (B, D) L2-normalized cloth embeddings
        body_vecs   : (B, 7) raw body measurement vectors (for debiasing)
 
        Returns: (loss, current_temperature)
        """
        B      = body_emb.size(0)
        t      = self.temperature
        logits = torch.matmul(body_emb, cloth_emb.T) / t   # (B, B)
 
        # ── 1. False-negative masking ────────────────────────────────
        fn_mask = self._false_negative_mask(body_vecs)
        # Replace false-negative positions with -inf so they don't
        # contribute to the softmax denominator
        logits_debiased = logits.masked_fill(fn_mask, float("-inf"))
 
        # ── 2. Semi-hard negative emphasis ──────────────────────────
        if self.hard_neg_weight > 0:
            with torch.no_grad():
                # Identify negatives that are almost-positives (hard)
                eye    = torch.eye(B, device=body_emb.device).bool()
                neg_mask = ~eye & ~fn_mask
                # Normalize to (0, 1) within each row
                neg_logits  = logits.masked_fill(~neg_mask, float("-inf"))
                hardness    = torch.softmax(neg_logits, dim=1)   # (B, B)
                # Boost hard negatives slightly — clamped to avoid explosion
                boost = (self.hard_neg_weight * hardness).clamp(max=0.5)
            logits_debiased = logits_debiased + boost * neg_mask.float()
 
        # ── 3. Bidirectional cross-entropy ───────────────────────────
        labels   = torch.arange(B, device=body_emb.device)
        loss_b2c = F.cross_entropy(logits_debiased,   labels)
        loss_c2b = F.cross_entropy(logits_debiased.T, labels)
        loss     = (loss_b2c + loss_c2b) / 2
 
        return loss, t.item()
 
 
# ================================================================
# COLLAPSE DIAGNOSTIC  (call during eval, not during training)
# ================================================================
 
@torch.no_grad()
def check_embedding_collapse(body_embs: torch.Tensor,
                             cloth_embs: torch.Tensor) -> dict:
    """
    Detects embedding collapse — the root cause of high training
    accuracy but near-zero Recall@1.
 
    Returns dict with:
      body_std       : mean std across embedding dims (want > 0.1)
      cloth_std      : same for clothing embeddings
      random_sim     : avg cosine similarity of random cloth pairs
                       (want < 0.1; > 0.3 = collapse confirmed)
      verdict        : 'OK' | 'WARNING' | 'COLLAPSED'
    """
    body_std  = body_embs.std(dim=0).mean().item()
    cloth_std = cloth_embs.std(dim=0).mean().item()
 
    n   = min(200, cloth_embs.size(0))
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
 
    return dict(body_std=body_std, cloth_std=cloth_std,
                random_sim=random_sim, verdict=verdict)
 
