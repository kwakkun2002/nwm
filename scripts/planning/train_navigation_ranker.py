import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.evaluation.planning.navigation_ranker import NavigationRanker


def split_by_group(groups, val_fraction, seed):
    rng = random.Random(seed)
    unique_groups = sorted(set(groups.tolist()))
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(len(unique_groups) * val_fraction))) if len(unique_groups) > 1 else 0
    val_groups = set(unique_groups[:n_val])
    is_val = torch.tensor([int(group) in val_groups for group in groups], dtype=torch.bool)
    return ~is_val, is_val


def pairwise_accuracy(pred, target, groups):
    total = 0
    correct = 0
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if mask.sum() < 2:
            continue
        pred_g = pred[mask]
        target_g = target[mask]
        i, j = torch.triu_indices(pred_g.numel(), pred_g.numel(), offset=1)
        target_order = target_g[i] < target_g[j]
        pred_order = pred_g[i] < pred_g[j]
        valid = target_g[i] != target_g[j]
        if valid.any():
            correct += (target_order[valid] == pred_order[valid]).sum().item()
            total += valid.sum().item()
    return correct / total if total else 0.0


def pairwise_rank_loss(pred, target, groups, margin):
    losses = []
    for group in torch.unique(groups):
        mask = groups == group
        if mask.sum() < 2:
            continue
        pred_g = pred[mask]
        target_g = target[mask]
        i, j = torch.triu_indices(pred_g.numel(), pred_g.numel(), offset=1, device=pred.device)
        sign = torch.sign(target_g[j] - target_g[i])
        valid = sign != 0
        if valid.any():
            diff = pred_g[j] - pred_g[i]
            losses.append(F.relu(margin - sign[valid] * diff[valid]).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def evaluate(model, features, targets_norm, targets_raw, groups, batch_size, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            preds.append(model(features[start:start + batch_size].to(device)).cpu())
    pred_norm = torch.cat(preds, dim=0)
    mse = F.mse_loss(pred_norm, targets_norm).item()
    pair_acc = pairwise_accuracy(pred_norm, targets_raw, groups)
    return mse, pair_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="artifacts/bulk/planning_ranker/checkpoints/navigation_ranker.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairwise_weight", type=float, default=0.2)
    parser.add_argument("--pairwise_margin", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    features = payload["features"].float()
    targets = payload["targets"].float()
    groups = payload["groups"].long()

    train_mask, val_mask = split_by_group(groups, args.val_fraction, args.seed)
    feature_mean = features[train_mask].mean(dim=0)
    feature_std = features[train_mask].std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = targets[train_mask].mean()
    target_std = targets[train_mask].std(unbiased=False).clamp_min(1e-6)

    features_norm = (features - feature_mean) / feature_std
    targets_norm = (targets - target_mean) / target_std

    train_dataset = TensorDataset(features_norm[train_mask], targets_norm[train_mask], targets[train_mask], groups[train_mask])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model_config = {
        "input_dim": int(features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
    }
    model = NavigationRanker(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_features, batch_targets_norm, batch_targets_raw, batch_groups in train_loader:
            batch_features = batch_features.to(device)
            batch_targets_norm = batch_targets_norm.to(device)
            batch_targets_raw = batch_targets_raw.to(device)
            batch_groups = batch_groups.to(device)
            pred = model(batch_features)
            mse = F.mse_loss(pred, batch_targets_norm)
            rank = pairwise_rank_loss(pred, batch_targets_raw, batch_groups, args.pairwise_margin)
            loss = mse + args.pairwise_weight * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_features.shape[0]
            total_items += batch_features.shape[0]

        train_mse, train_pair = evaluate(
            model,
            features_norm[train_mask],
            targets_norm[train_mask],
            targets[train_mask],
            groups[train_mask],
            args.batch_size,
            device,
        )
        if val_mask.any():
            val_mse, val_pair = evaluate(
                model,
                features_norm[val_mask],
                targets_norm[val_mask],
                targets[val_mask],
                groups[val_mask],
                args.batch_size,
                device,
            )
        else:
            val_mse, val_pair = train_mse, train_pair
        history.append({
            "epoch": epoch,
            "loss": total_loss / max(total_items, 1),
            "train_mse": train_mse,
            "train_pairwise_accuracy": train_pair,
            "val_mse": val_mse,
            "val_pairwise_accuracy": val_pair,
        })
        if val_mse < best_val:
            best_val = val_mse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} train_mse={train_mse:.4f} train_pair={train_pair:.3f} "
                f"val_mse={val_mse:.4f} val_pair={val_pair:.3f}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": best_state,
        "model_config": model_config,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "dataset_metadata": payload.get("metadata", {}),
        "history": history,
    }
    torch.save(checkpoint, output)
    summary = {
        "checkpoint": str(output),
        "best_val_mse": best_val,
        "final": history[-1],
        "num_train": int(train_mask.sum()),
        "num_val": int(val_mask.sum()),
    }
    with output.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
