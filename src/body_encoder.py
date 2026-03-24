import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BodyEncoder(nn.Module):

    def __init__(self, input_dim=7, embedding_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # 🔥 Feature normalization (learnable optional later)
        self.register_buffer("eps", torch.tensor(1e-6))

        # ================= IMPROVED MLP =================
        self.h_meas = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # 🔥 CRITICAL: normalize input features
    def normalize_input(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + self.eps)

    def forward(self, x):

        # 🔥 normalize body features FIRST
        x = self.normalize_input(x)

        embedding = self.h_meas(x)

        # 🔥 normalize embedding (for contrastive space)
        embedding = F.normalize(embedding, dim=1)

        return embedding


    # ================= INFERENCE =================

    def get_embedding(self, measurements):

        squeeze = (measurements.ndim == 1)
        if squeeze:
            measurements = measurements[np.newaxis, :]

        tensor = torch.tensor(measurements, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            emb = self.forward(tensor)

        result = emb.cpu().numpy()
        return result[0] if squeeze else result


# ================= FEATURE EXTRACTION =================

class BodyMeasurementExtractor:

    def extract(self, image_path):

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

        lm = result.pose_landmarks.landmark

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

        shoulder_mid = mid(pts["left_shoulder"], pts["right_shoulder"])
        hip_mid      = mid(pts["left_hip"], pts["right_hip"])
        ankle_mid    = mid(pts["left_ankle"], pts["right_ankle"])

        height     = dist(pts["nose"], ankle_mid)
        shoulder_w = dist(pts["left_shoulder"], pts["right_shoulder"])
        hip_w      = dist(pts["left_hip"], pts["right_hip"])

        bust   = shoulder_w * 1.3
        waist  = hip_w * 0.85

        shoulder_hip_ratio = shoulder_w / (hip_w + 1e-6)
        torso_h            = dist(shoulder_mid, hip_mid)

        return np.array([
            height, bust, waist, hip_w,
            shoulder_w, shoulder_hip_ratio, torso_h
        ], dtype=np.float32)


# ================= TEST =================

if __name__ == "__main__":

    print("=" * 55)
    print("  BodyEncoder — FINAL OPTIMIZED VERSION")
    print("=" * 55)

    model = BodyEncoder(input_dim=7, embedding_dim=128)

    dummy = torch.randn(4, 7)
    output = model(dummy)

    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"L2 norms     : {output.norm(dim=1).tolist()}")