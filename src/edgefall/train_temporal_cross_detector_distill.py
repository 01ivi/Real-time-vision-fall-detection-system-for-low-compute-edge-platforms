"""Distill a small GeometryTCN across detector feature domains.

Teacher path:
    yolo26m features -> large teacher TCN -> soft logits

Student path:
    yolo26n features -> small student TCN -> learned logits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edgefall.models.temporal_tcn import build_geometry_tcn
from edgefall.utils.config import ensure_dir, load_config
from edgefall.utils.deps import require_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Student config")
    parser.add_argument("--student-train-manifest", required=True, help="Feature JSONL extracted by the student detector, e.g. yolo26n")
    parser.add_argument("--teacher-train-manifest", required=True, help="Feature JSONL extracted by the teacher detector, e.g. yolo26m")
    parser.add_argument("--student-val-manifest", required=True)
    parser.add_argument("--teacher-val-manifest", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="0=CE only, 1=KD only")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--strict-pairing", action="store_true", help="Fail if any student row has no teacher row")
    return parser.parse_args()


def parse_csv_labels(value, default: tuple[str, ...]) -> set[str]:
    if value is None:
        return set(default)
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(item) for item in value}


def event_key(item: dict[str, Any]) -> tuple:
    video = str(item.get("video") or item.get("path"))
    label = str(item.get("label"))
    start = round(float(item.get("start", 0.0) or 0.0), 4)
    end = round(float(item.get("end", 0.0) or 0.0), 4)
    start_frame = int(item.get("start_frame", 0) or 0)
    end_frame = item.get("end_frame")
    end_frame = None if end_frame in (None, "") else int(end_frame)
    return video, label, start, end, start_frame, end_frame


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pad_or_trim(features: list, clip_len: int, geometry_dim: int) -> list:
    if len(features) < clip_len:
        pad_item = features[-1] if features else [0.0] * geometry_dim
        features = features + [pad_item] * (clip_len - len(features))
    return features[:clip_len]


class CrossDetectorFeatureDataset:
    def __init__(
        self,
        student_manifest: str,
        teacher_manifest: str,
        labels: list[str],
        clip_len: int,
        geometry_dim: int,
        strict_pairing: bool = False,
    ) -> None:
        torch = require_torch()
        self.torch = torch
        self.labels = {name: idx for idx, name in enumerate(labels)}
        self.clip_len = clip_len
        self.geometry_dim = geometry_dim
        teacher_rows = read_jsonl(teacher_manifest)
        teacher_by_key = {event_key(row): row for row in teacher_rows}
        self.pairs = []
        missing = 0
        for student_row in read_jsonl(student_manifest):
            teacher_row = teacher_by_key.get(event_key(student_row))
            if teacher_row is None:
                missing += 1
                if strict_pairing:
                    raise ValueError(f"No teacher feature row for key={event_key(student_row)}")
                continue
            if "features" not in student_row or "features" not in teacher_row:
                raise ValueError("Both student and teacher manifests must contain 'features'.")
            if str(student_row["label"]) != str(teacher_row["label"]):
                raise ValueError(f"Label mismatch for key={event_key(student_row)}")
            self.pairs.append((student_row, teacher_row))
        if not self.pairs:
            raise ValueError("No paired rows found. Check that both manifests are generated from the same source manifest.")
        print(
            f"Paired {len(self.pairs)} rows from student={student_manifest} "
            f"and teacher={teacher_manifest}; missing={missing}"
        )

    @property
    def records(self) -> list[dict[str, Any]]:
        return [student for student, _ in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        student_row, teacher_row = self.pairs[idx]
        student_features = pad_or_trim(student_row["features"], self.clip_len, self.geometry_dim)
        teacher_features = pad_or_trim(teacher_row["features"], self.clip_len, self.geometry_dim)
        x_student = self.torch.tensor(student_features, dtype=self.torch.float32)
        x_teacher = self.torch.tensor(teacher_features, dtype=self.torch.float32)
        y = self.torch.tensor(self.labels[str(student_row["label"])], dtype=self.torch.long)
        return x_student, x_teacher, y


def build_from_config(cfg: dict, num_classes: int):
    return build_geometry_tcn(
        geometry_dim=int(cfg["data"]["geometry_dim"]),
        num_classes=num_classes,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    )


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


def build_sampler(torch, dataset: CrossDetectorFeatureDataset, train_cfg: dict, generator=None):
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


def evaluate(student, loader, device: str, criterion, labels: list[str], positive_labels: set[str]):
    torch = require_torch()
    student.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    positive_indices = {idx for idx, label in enumerate(labels) if label in positive_labels}
    tp = fp = fn = 0
    with torch.no_grad():
        for x_student, _, y in loader:
            x_student = x_student.to(device)
            y = y.to(device)
            logits = student(x_student)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=-1)
            loss_sum += float(loss.item()) * y.numel()
            correct += int((pred == y).sum().item())
            total += y.numel()
            for true_idx, pred_idx in zip(y.cpu().tolist(), pred.cpu().tolist()):
                true_positive = int(true_idx) in positive_indices
                pred_positive = int(pred_idx) in positive_indices
                if true_positive and pred_positive:
                    tp += 1
                elif (not true_positive) and pred_positive:
                    fp += 1
                elif true_positive and (not pred_positive):
                    fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "positive_tp": tp,
        "positive_fp": fp,
        "positive_fn": fn,
    }


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

    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=args.device)
    if teacher_ckpt["labels"] != labels:
        raise ValueError("Teacher and student labels must be identical for logit distillation.")
    teacher = build_from_config(teacher_ckpt["config"], num_classes=len(labels)).to(args.device)
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = build_from_config(cfg, num_classes=len(labels)).to(args.device)

    train_set = CrossDetectorFeatureDataset(
        args.student_train_manifest,
        args.teacher_train_manifest,
        labels=labels,
        clip_len=int(cfg["data"]["clip_len"]),
        geometry_dim=int(cfg["data"]["geometry_dim"]),
        strict_pairing=args.strict_pairing,
    )
    val_set = CrossDetectorFeatureDataset(
        args.student_val_manifest,
        args.teacher_val_manifest,
        labels=labels,
        clip_len=int(cfg["data"]["clip_len"]),
        geometry_dim=int(cfg["data"]["geometry_dim"]),
        strict_pairing=args.strict_pairing,
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

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    class_weights = build_class_weights(torch, labels, train_cfg, args.device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    kd_loss = torch.nn.KLDivLoss(reduction="batchmean")
    alpha = float(args.distill_alpha)
    temperature = float(args.temperature)
    best_key = None

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        student.train()
        running = 0.0
        seen = 0
        for x_student, x_teacher, y in train_loader:
            x_student = x_student.to(args.device)
            x_teacher = x_teacher.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(x_student)
            with torch.no_grad():
                teacher_logits = teacher(x_teacher)
            hard_loss = ce_loss(student_logits, y)
            soft_loss = kd_loss(
                torch.log_softmax(student_logits / temperature, dim=-1),
                torch.softmax(teacher_logits / temperature, dim=-1),
            ) * (temperature * temperature)
            loss = (1.0 - alpha) * hard_loss + alpha * soft_loss
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * y.numel()
            seen += y.numel()

        metrics = evaluate(student, val_loader, args.device, ce_loss, labels, positive_labels)
        print(
            f"epoch={epoch:03d} train_loss={running / max(seen, 1):.4f} "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} "
            f"fall_precision={metrics['positive_precision']:.4f} "
            f"fall_recall={metrics['positive_recall']:.4f} "
            f"fall_f1={metrics['positive_f1']:.4f}"
        )
        last = {
            "model": student.state_dict(),
            "labels": labels,
            "config": cfg,
            "epoch": epoch,
            "val": metrics,
            "positive_labels": sorted(positive_labels),
            "teacher_checkpoint": args.teacher_checkpoint,
            "distill_alpha": alpha,
            "temperature": temperature,
            "student_feature_domain": args.student_train_manifest,
            "teacher_feature_domain": args.teacher_train_manifest,
            "checkpoint_metric": checkpoint_metric,
            "min_checkpoint_recall": min_checkpoint_recall,
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
