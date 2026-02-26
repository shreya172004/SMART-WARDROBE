"""
body_encoder.py
═══════════════════════════════════════════════════════════════

  "For body shape, we accompany estimated 3D parameters (10-D)
   with vital statistics (4-D). Each is first reduced into a
   lower dimensional space with learned projection functions
   h_smpl, h_meas. Then the reduced features are concatenated
   as the representation xb for body shape and forwarded into
   the joint embedding by f_body."

   "All dimensionality reduction functions h_attr, h_cnn,
    h_smpl, h_meas are 2-layer MLPs, and the embedding
    functions f_cloth and f_body are single fully connected
    layers."
    
  We use 7 body measurements from MediaPipe as input
  instead of SMPL(10D) + vital_stats(4D).
  
Rules:
  ✓ input_dim  = 7   (body measurements)
  ✓ output_dim = 128 (embedding)
  ✓ Does NOT touch dataset
  ✓ Does NOT contain training logic
  ✓ Only produces embeddings

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════
# THE 7 INPUT FEATURES
# ═══════════════════════════════════════════════════════════════
#
#  Index | Feature               | Source
#  ──────────────────────────────────────────────────────────
#    0   | height (normalized)   | nose → ankle distance
#    1   | bust  (normalized)    | shoulder width × 1.3
#    2   | waist (normalized)    | hip width × 0.85
#    3   | hip   (normalized)    | left_hip → right_hip
#    4   | shoulder_width        | left_shoulder → right_shoulder
#    5   | shoulder_hip_ratio    | shoulder / hip
#    6   | torso_height          | shoulder_mid → hip_mid
#
# ═══════════════════════════════════════════════════════════════


class BodyEncoder(nn.Module):
    """
    Body Measurement Encoder

    Maps 7-dim body measurement vector → 128-dim embedding.

    Implements h_meas + f_body :

        h_meas : 2-layer MLP (dimensionality reduction)
        f_body : single FC layer (projects to embedding space)
        L2 norm: constrains to unit hypersphere (Section 3.3)

    Full Architecture:
        Input (7)
          ↓
        ┌─────────────────────────────┐
        │  h_meas  (2-layer MLP)      │
        │  Linear(7→32) + BN + ReLU  │
        │  Linear(32→64) + BN + ReLU │
        └─────────────────────────────┘
          ↓
        ┌─────────────────────────────┐
        │  f_body  (single FC layer)  │
        │  Linear(64→128)             │
        └─────────────────────────────┘
          ↓
        L2 Normalize (unit hypersphere)
          ↓
        Output (128-dim continuous embedding)
    """

    def __init__(self, input_dim=7, embedding_dim=128):
        super(BodyEncoder, self).__init__()

        self.input_dim     = input_dim       # 7 body measurements
        self.embedding_dim = embedding_dim   # 128-dim output

        self.h_meas = nn.Sequential(

            # Layer 1: 7 → 32
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Layer 2: 32 → 64
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.f_body = nn.Linear(64, embedding_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x : Tensor (batch_size, 7) — standardized measurements

        Returns:
            embedding : Tensor (batch_size, 128)
                        L2-normalized, on unit hypersphere
                        CONTINUOUS — unique per person
        """
        # h_meas: reduce to compact representation
        reduced   = self.h_meas(x)           # (batch, 7) → (batch, 64)

        # f_body: project to embedding space
        embedding = self.f_body(reduced)     # (batch, 64) → (batch, 128)

        # L2 normalize — unit hypersphere (paper Section 3.3)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding                     # (batch, 128)

    def get_embedding(self, measurements):
        """
        Convenience: numpy → numpy embedding.

        Args:
            measurements : numpy (7,) or (N, 7)

        Returns:
            embedding    : numpy (128,) or (N, 128)
        """
        squeeze = (measurements.ndim == 1)
        if squeeze:
            measurements = measurements[np.newaxis, :]

        tensor = torch.tensor(measurements, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            emb = self.forward(tensor)

        result = emb.numpy()
        return result[0] if squeeze else result


# BODY MEASUREMENT EXTRACTOR
# Extracts the 7 features from a person image

class BodyMeasurementExtractor:
    """
    Extracts 7-dim body measurement vector from an image
    using MediaPipe Pose — matches BodyEncoder input_dim=7.
    """

    def extract(self, image_path):
        """
        Returns numpy (7,) or zeros if no pose detected.
        """
        import cv2
        import mediapipe as mp

        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(7, dtype=np.float32)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        ) as pose:
            result = pose.process(image_rgb)

        if not result.pose_landmarks:
            return np.zeros(7, dtype=np.float32)

        lm  = result.pose_landmarks.landmark

        pts = {
            "nose":           (lm[0].x,  lm[0].y),
            "left_shoulder":  (lm[11].x, lm[11].y),
            "right_shoulder": (lm[12].x, lm[12].y),
            "left_hip":       (lm[23].x, lm[23].y),
            "right_hip":      (lm[24].x, lm[24].y),
            "left_ankle":     (lm[27].x, lm[27].y),
            "right_ankle":    (lm[28].x, lm[28].y),
        }

        def dist(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        def mid(p1, p2):
            return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

        shoulder_mid       = mid(pts["left_shoulder"], pts["right_shoulder"])
        hip_mid            = mid(pts["left_hip"],      pts["right_hip"])
        ankle_mid          = mid(pts["left_ankle"],    pts["right_ankle"])

        height             = dist(pts["nose"],            ankle_mid)
        shoulder_w         = dist(pts["left_shoulder"],   pts["right_shoulder"])
        hip_w              = dist(pts["left_hip"],        pts["right_hip"])
        bust               = shoulder_w * 1.3
        waist              = hip_w * 0.85
        shoulder_hip_ratio = shoulder_w / (hip_w + 1e-6)
        torso_h            = dist(shoulder_mid, hip_mid)

        return np.array([
            height, bust, waist, hip_w,
            shoulder_w, shoulder_hip_ratio, torso_h
        ], dtype=np.float32)

# QUICK TEST

if __name__ == "__main__":
    print("=" * 55)
    print("  BodyEncoder — ViBE Paper Architecture")
    print("=" * 55)

    model     = BodyEncoder(input_dim=7, embedding_dim=128)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f"\n  input_dim     : 7   (body measurements)")
    print(f"  h_meas        : 2-layer MLP  7 → 32 → 64")
    print(f"  f_body        : single FC   64 → 128")
    print(f"  output_dim    : 128 (L2 normalized)")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}")

    dummy  = torch.randn(4, 7)
    output = model(dummy)

    print(f"\n  Input  shape  : {dummy.shape}")
    print(f"  Output shape  : {output.shape}")
    print(f"  L2 norms      : {output.norm(dim=1).tolist()}")
    print(f"  (All should be exactly 1.0)\n")
    print(f"   BodyEncoder ready — 128-dim continuous embeddings")
