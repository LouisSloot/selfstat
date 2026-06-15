"""Appearance embedding for tracklet crops.

Primary backbone is DINOv2 (ViT-S/14) via torch.hub — the project's chosen re-ID
feature extractor (strong, general, self-supervised; far better than the old
OSNet-x0_25). If the hub model can't be fetched, we fall back to a torchvision
ResNet50 so the pipeline still runs. A tracklet is summarized by mean-pooling the
L2-normalized per-crop features, which averages out occlusion/pose noise.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Embedder:
    def __init__(self, device, input_size=224, batch_size=32):
        self.device = device
        self.input_size = input_size
        self.batch_size = batch_size
        self.model, self.dim = self._load_backbone()
        self.model.eval().to(device)
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    def _load_backbone(self):
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14",
                trust_repo=True, verbose=False,
            )
            print("[embedder] backbone = DINOv2 ViT-S/14 (dim 384)")
            return model, 384
        except Exception as e:  # network/git/hub issues -> still produce features
            print(f"[embedder] DINOv2 unavailable ({type(e).__name__}: {e}); "
                  f"falling back to torchvision ResNet50")
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Identity()
            return model, 2048

    def _preprocess(self, crops):
        batch = []
        for crop in crops:
            if crop is None or crop.size == 0:
                crop = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.input_size, self.input_size))
            batch.append(rgb)
        arr = np.stack(batch).astype(np.float32) / 255.0  # N,H,W,3
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)  # N,3,H,W
        return (t - self.mean) / self.std

    @torch.no_grad()
    def embed(self, crops):
        """Mean-pooled, L2-normalized embedding (np.ndarray [dim]) for a crop set."""
        if not crops:
            return np.zeros(self.dim, dtype=np.float32)
        feats = []
        for i in range(0, len(crops), self.batch_size):
            t = self._preprocess(crops[i:i + self.batch_size])
            f = self.model(t)
            feats.append(F.normalize(f, p=2, dim=1).cpu())
        pooled = torch.cat(feats, dim=0).mean(dim=0)
        pooled = F.normalize(pooled, p=2, dim=0)
        return pooled.numpy().astype(np.float32)
