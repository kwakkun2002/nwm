import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms.action import get_action_torch


DEFAULT_DINO_WEIGHTS = "weights/pretrained/dino/dino_vitbase16_pretrain.pth"
DINO_SOURCE_DIR = Path(__file__).resolve().parents[3] / "third_party" / "facebookresearch_dino_main"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class NavigationRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, depth=2, dropout=0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DinoFeatureExtractor(nn.Module):
    def __init__(self, weights_path=DEFAULT_DINO_WEIGHTS, arch="vit_base", patch_size=16, image_size=224):
        super().__init__()
        self.image_size = image_size
        if str(DINO_SOURCE_DIR) not in sys.path:
            sys.path.insert(0, str(DINO_SOURCE_DIR))

        import vision_transformer as vits

        if arch == "vit_base":
            model = vits.vit_base(patch_size=patch_size, num_classes=0)
            self.feature_dim = 768
        elif arch == "vit_small":
            model = vits.vit_small(patch_size=patch_size, num_classes=0)
            self.feature_dim = 384
        else:
            raise ValueError(f"Unsupported DINO arch: {arch}")

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINO weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "teacher" in state:
            state = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in state["teacher"].items()
            }
        model.load_state_dict(state, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model

    def preprocess(self, images):
        images = images.float()
        if images.min() < -0.05:
            images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        mean = IMAGENET_MEAN.to(images)
        std = IMAGENET_STD.to(images)
        return (images - mean) / std

    @torch.no_grad()
    def forward(self, images, batch_size=64):
        self.model.float()
        model_device = next(self.model.parameters()).device
        feats = []
        for start in range(0, images.shape[0], batch_size):
            batch = self.preprocess(images[start:start + batch_size].to(model_device))
            batch = batch.to(device=model_device, dtype=torch.float32)
            if batch.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    feat = self.model(batch)
            else:
                feat = self.model(batch)
            feats.append(F.normalize(feat.float(), dim=-1).cpu())
        return torch.cat(feats, dim=0)


def action_feature_dim():
    return 8


def compute_action_features(deltas, action_stats):
    pred_actions = get_action_torch(deltas[:, :, :2], action_stats).float()
    zero = torch.zeros(pred_actions.shape[0], 1, 2, device=pred_actions.device, dtype=pred_actions.dtype)
    steps = torch.diff(torch.cat((zero, pred_actions[:, :, :2]), dim=1), dim=1)
    step_norm = torch.norm(steps, dim=-1)
    if steps.shape[1] > 1:
        step_change = torch.diff(steps, dim=1)
        smoothness = torch.norm(step_change, dim=-1).mean(dim=1)
    else:
        smoothness = torch.zeros(pred_actions.shape[0], device=pred_actions.device, dtype=pred_actions.dtype)

    final_xy = pred_actions[:, -1, :2]
    total_yaw = deltas[:, :, 2].sum(dim=1, keepdim=True)
    path_len = step_norm.sum(dim=1, keepdim=True)
    final_norm = torch.norm(final_xy, dim=-1, keepdim=True)
    mean_step = step_norm.mean(dim=1, keepdim=True)
    max_step = step_norm.max(dim=1, keepdim=True).values
    smoothness = smoothness.unsqueeze(1)
    return torch.cat((final_xy, total_yaw, path_len, final_norm, mean_step, max_step, smoothness), dim=1)


def build_ranker_features(pred_images, goal_images, deltas, dino, action_stats, dino_batch_size=64):
    pred_feats = dino(pred_images, batch_size=dino_batch_size)
    goal_feats = dino(goal_images, batch_size=dino_batch_size)
    action_feats = compute_action_features(deltas.detach().cpu(), {
        key: value.detach().cpu() if torch.is_tensor(value) else torch.tensor(value)
        for key, value in action_stats.items()
    })
    return torch.cat((pred_feats, goal_feats, (pred_feats - goal_feats).abs(), action_feats), dim=1)


def load_ranker_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    model = NavigationRanker(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint


def score_with_ranker(features, model, checkpoint, device):
    feature_mean = checkpoint["feature_mean"].to(features)
    feature_std = checkpoint["feature_std"].to(features).clamp_min(1e-6)
    target_mean = checkpoint["target_mean"].to(features)
    target_std = checkpoint["target_std"].to(features).clamp_min(1e-6)
    norm_features = (features - feature_mean) / feature_std
    with torch.no_grad():
        pred = model(norm_features.to(device)).cpu()
    return pred * target_std.cpu() + target_mean.cpu()
