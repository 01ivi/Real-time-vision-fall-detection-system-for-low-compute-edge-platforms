"""Train a lightweight geometry TCN action-stage classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgefall.data.dataset import GeometryFeatureDataset
from edgefall.models.temporal_tcn import build_geometry_tcn
from edgefall.utils.config import ensure_dir, load_config
from edgefall.utils.deps import require_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint used to initialize the TCN before fine-tuning")
    return parser.parse_args()


def parse_csv_labels(value, default: tuple[str, ...]) -> set[str]:
    if value is None:
        return set(default)
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(item) for item in value}


def build_class_weights(torch, labels: list[str], train_cfg: dict, device: str):
    weights_cfg = train_cfg.get("class_weights")
    if weights_cfg is None:
        return None
    if isinstance(weights_cfg, list):
        if len(weights_cfg) != len(labels):
            raise ValueError(f"class_weights list length {len(weights_cfg)} != labels length {len(labels)}")
        weights = [float(value) for value in weights_cfg]
    elif isinstance(weights_cfg, dict):
        weights = [float(weights_cfg.get(label, 1.0)) for label in labels]
    else:
        raise TypeError("train.class_weights must be null, a list, or a mapping of label -> weight.")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_sampler(torch, dataset: GeometryFeatureDataset, train_cfg: dict, generator=None):
    sample_weights_cfg = train_cfg.get("sample_weights")
    if not sample_weights_cfg:
        return None
    weights = [float(sample_weights_cfg.get(record["label"], 1.0)) for record in dataset.records]
    multiplier = float(train_cfg.get("sampler_epoch_multiplier", 1.0))
    num_samples = max(1, int(round(len(weights) * multiplier)))
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def evaluate(model, loader, device: str, criterion, labels: list[str], positive_labels: set[str]):
    torch = require_torch()
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    positive_indices = {idx for idx, label in enumerate(labels) if label in positive_labels}
    tp = fp = fn = 0
    per_label_total = {label: 0 for label in labels}
    per_label_correct = {label: 0 for label in labels}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=-1)
            loss_sum += float(loss.item()) * y.numel()
            correct += int((pred == y).sum().item())
            total += y.numel()
            for true_idx, pred_idx in zip(y.cpu().tolist(), pred.cpu().tolist()):
                true_label = labels[int(true_idx)]
                pred_label = labels[int(pred_idx)]
                per_label_total[true_label] += 1
                if true_idx == pred_idx:
                    per_label_correct[true_label] += 1
                true_positive = int(true_idx) in positive_indices
                pred_positive = int(pred_idx) in positive_indices
                if true_positive and pred_positive:
                    tp += 1
                elif (not true_positive) and pred_positive:
                    fp += 1
                elif true_positive and (not pred_positive):
                    fn += 1
    positive_precision = tp / max(tp + fp, 1)
    positive_recall = tp / max(tp + fn, 1)
    positive_f1 = 2 * positive_precision * positive_recall / max(positive_precision + positive_recall, 1e-12)
    metrics = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "positive_precision": positive_precision,
        "positive_recall": positive_recall,
        "positive_f1": positive_f1,
        "positive_tp": tp,
        "positive_fp": fp,
        "positive_fn": fn,
    }
    for label in positive_labels:
        if label in per_label_total:
            metrics[f"{label}_recall"] = per_label_correct[label] / max(per_label_total[label], 1)
    return metrics


def main() -> None:
    args = parse_args()
    torch = require_torch()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    labels = cfg["data"]["labels"]
    train_cfg = cfg["train"]
    positive_labels = parse_csv_labels(train_cfg.get("positive_labels"), default=("fall", "fallen"))
    output_dir = ensure_dir(args.output)
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "positive_recall"))
    min_checkpoint_recall = float(train_cfg.get("min_checkpoint_recall", 0.0))

    train_set = GeometryFeatureDataset(
        args.train_manifest,
        labels=labels,
        clip_len=int(cfg["data"]["clip_len"]),
        geometry_dim=int(cfg["data"]["geometry_dim"]),
    )
    val_set = GeometryFeatureDataset(
        args.val_manifest,
        labels=labels,
        clip_len=int(cfg["data"]["clip_len"]),
        geometry_dim=int(cfg["data"]["geometry_dim"]),
    )
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    sampler = build_sampler(torch, train_set, train_cfg, generator=sampler_generator)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        generator=loader_generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = build_geometry_tcn(
        geometry_dim=int(cfg["data"]["geometry_dim"]),
        num_classes=len(labels),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(args.device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        if checkpoint.get("labels") != labels:
            raise ValueError(f"Checkpoint labels do not match config labels: {args.init_checkpoint}")
        model.load_state_dict(checkpoint["model"])
        print(f"Initialized model from {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    class_weights = build_class_weights(torch, labels, train_cfg, args.device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_key = None

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * y.numel()
            seen += y.numel()

        metrics = evaluate(model, val_loader, args.device, criterion, labels, positive_labels)
        print(
            f"epoch={epoch:03d} train_loss={running / max(seen, 1):.4f} "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} "
            f"fall_precision={metrics['positive_precision']:.4f} "
            f"fall_recall={metrics['positive_recall']:.4f} "
            f"fall_f1={metrics['positive_f1']:.4f}"
        )
        last = {
            "model": model.state_dict(),
            "labels": labels,
            "config": cfg,
            "epoch": epoch,
            "val": metrics,
            "positive_labels": sorted(positive_labels),
            "checkpoint_metric": checkpoint_metric,
            "min_checkpoint_recall": min_checkpoint_recall,
            "init_checkpoint": args.init_checkpoint,
        }
        if checkpoint_metric == "positive_recall":
            key = (
                metrics["positive_recall"],
                metrics["positive_f1"],
                metrics["positive_precision"],
                metrics["acc"],
                -metrics["loss"],
            )
        elif checkpoint_metric == "high_recall_f1":
            if metrics["positive_recall"] < min_checkpoint_recall:
                continue
            key = (
                metrics["positive_f1"],
                metrics["positive_precision"],
                metrics["positive_recall"],
                metrics["acc"],
                -metrics["loss"],
            )
        elif checkpoint_metric == "positive_f1":
            key = (
                metrics["positive_f1"],
                metrics["positive_recall"],
                metrics["positive_precision"],
                metrics["acc"],
                -metrics["loss"],
            )
        else:
            raise ValueError(f"Unsupported train.checkpoint_metric: {checkpoint_metric}")

        if best_key is None or key > best_key:
            best_key = key
            torch.save(last, output_dir / "best.pt")

    if best_key is None:
        raise RuntimeError(
            f"No checkpoint satisfied train.checkpoint_metric={checkpoint_metric!r} "
            f"with min_checkpoint_recall={min_checkpoint_recall}."
        )


if __name__ == "__main__":
    main()
