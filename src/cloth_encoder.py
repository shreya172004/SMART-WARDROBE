import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

cloth_transform = transforms.Compose([
    transforms.Resize((224, 224)),       
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       
        std =[0.229, 0.224, 0.225]       
    )
])

class ClothEncoder(nn.Module):
    
    def __init__(self, embedding_dim=128, freeze_backbone=True):
        super(ClothEncoder, self).__init__()

        self.embedding_dim = embedding_dim

        resnet      = models.resnet50(pretrained=True)
        backbone_dim = resnet.fc.in_features   

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
      
        if freeze_backbone:
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers):
                if i < 7:   # freeze layers 0-6, train 7-8
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.h_cnn = nn.Sequential(

            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.f_cloth = nn.Linear(256, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize only the new layers, not pretrained backbone."""
        for m in [self.h_cnn, self.f_cloth]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.BatchNorm1d):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x)
  
        features  = self.backbone(x)            
        features  = features.view(features.size(0), -1)  

        reduced   = self.h_cnn(features)      
        embedding = self.f_cloth(reduced)        
      
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding                          

    def get_embedding(self, image_path):
        
        image  = Image.open(image_path).convert("RGB")
        tensor = cloth_transform(image).unsqueeze(0)  

        self.eval()
        with torch.no_grad():
            emb = self.forward(tensor)

        return emb.squeeze(0).numpy()

    def get_embedding_batch(self, image_paths):
       
        tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(cloth_transform(img))

        batch = torch.stack(tensors)  

        self.eval()
        with torch.no_grad():
            embs = self.forward(batch)

        return embs.numpy() 

if __name__ == "__main__":
    print("=" * 55)
    print("  ClothEncoder — ViBE Paper Architecture")
    print("=" * 55)

    model     = ClothEncoder(embedding_dim=128, freeze_backbone=True)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f"\n  Backbone      : ResNet50 (pretrained ImageNet)")
    print(f"  Backbone out  : 2048-dim")
    print(f"  h_cnn         : 2-layer MLP  2048 → 512 → 256")
    print(f"  f_cloth       : single FC    256  → 128")
    print(f"  output_dim    : 128 (L2 normalized)")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}")

    dummy  = torch.randn(4, 3, 224, 224)
    output = model(dummy)

    print(f"\n  Input  shape  : {dummy.shape}")
    print(f"  Output shape  : {output.shape}")
    print(f"  L2 norms      : {output.norm(dim=1).tolist()}")
    print(f"  (All should be exactly 1.0)\n")
    print(f"  ✓ ClothEncoder ready — 128-dim continuous embeddings")

