import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BodyEncoder(nn.Module):
    
    def __init__(self, input_dim=7, embedding_dim=128):
        super(BodyEncoder, self).__init__()

        self.input_dim     = input_dim       
        self.embedding_dim = embedding_dim   

        self.h_meas = nn.Sequential(

           nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

          
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.f_body = nn.Linear(64, embedding_dim)

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
        
        reduced   = self.h_meas(x)           
      
        embedding = self.f_body(reduced)     

        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding                   

    def get_embedding(self, measurements):
        
        squeeze = (measurements.ndim == 1)
        if squeeze:
            measurements = measurements[np.newaxis, :]

        tensor = torch.tensor(measurements, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            emb = self.forward(tensor)

        result = emb.numpy()
        return result[0] if squeeze else result


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

