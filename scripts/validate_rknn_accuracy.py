#!/usr/bin/env python3
"""Validate RKNN INT8 accuracy against PyTorch FP32 baseline.

Compares detector outputs (person boxes) and TCN outputs (logits/probs)
between the PyTorch FP32 reference and the RKNN INT8 models, on the
same test videos.  Produces a quantitative accuracy report.

This is the accuracy-preservation checkpoint — if the metrics degrade
significantly, the conversion parameters (calibration size, quant
scheme) should be adjusted before deployment.

Workflow:
  1. Run PyTorch FP32 inference on a test video → baseline events
  2. Run RKNN INT8 inference on the same video → candidate events
  3. Compare per-frame detection IoU, TCN class agreement, alarm overlap

Metrics reported:
  - Detector: mean IoU, detection count delta, box coordinate RMSE
  - TCN: logit cosine similarity, top-1 agreement rate, score correlation
  - End-to-end: alarm IoU matching, event-level precision/recall vs baseline

Example (on RK3588 board, running FP32 on CPU as reference)
------------------------------------------------------------
python3 scripts/validate_rknn_accuracy.py \
  --video /path/to/test_video.mp4 \
  --detector-pt checkpoint/yolo26m.pt \
  --detector-rknn runs/final/export/yolo26m_int8_imgsz960_modelzoo_sub100_rk3588.rknn \
  --temporal-pt runs/tcn_itw_mixed_finetune_yolo26m/best.pt \
  --temporal-rknn runs/final/export/temporal_head_int8_ln_addrelu_decomposed_cliprelu_b1_rk3588.rknn \
  --output-dir runs/rknn_validation

Or on x86_64 dev machine (RKNN-Toolkit2 supports eval/inference too):
  Use rknn.api.RKNN for accuracy checking (supports accuracy_analysis).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────


def _iou(a: list[float], b: list[float]) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(dot / (na * nb + 1e-8))


@dataclass
class ValidationReport:
    """Aggregated accuracy metrics."""

    # Detector
    det_mean_iou: float = 0.0
    det_count_delta_pct: float = 0.0  # % change in #detections
    det_box_rmse: float = 0.0  # pixel RMSE on box coords

    # TCN
    tcn_logit_cosine: float = 0.0  # cosine similarity of raw logits
    tcn_top1_agree: float = 0.0  # fraction where top-1 class matches
    tcn_score_corr: float = 0.0  # Pearson correlation of positive scores

    # End-to-end
    e2e_alarm_iou: float = 0.0  # temporal IoU of alarm events
    e2e_alarm_recall: float = 0.0  # fraction of PT alarms also found by RKNN

    frames_compared: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "RKNN Accuracy Validation Report",
            "=" * 60,
            f"Frames compared: {self.frames_compared}",
            "",
            "--- Detector (YOLO) ---",
            f"  Mean box IoU (PT vs RKNN):         {self.det_mean_iou:.4f}",
            f"  Detection count delta:              {self.det_count_delta_pct:+.1f}%",
            f"  Box coordinate RMSE (pixels):       {self.det_box_rmse:.2f}",
            "",
            "--- TCN (GeometryTCN) ---",
            f"  Logit cosine similarity:            {self.tcn_logit_cosine:.4f}",
            f"  Top-1 class agreement:              {self.tcn_top1_agree:.4f}",
            f"  Positive-score Pearson correlation: {self.tcn_score_corr:.4f}",
            "",
            "--- End-to-End Alarm ---",
            f"  Alarm temporal IoU (vs PT):         {self.e2e_alarm_iou:.4f}",
            f"  Alarm recall (vs PT as reference):  {self.e2e_alarm_recall:.4f}",
            "",
            "Assessment:",
        ]
        # Heuristic thresholds
        if (self.det_mean_iou > 0.90 and self.tcn_top1_agree > 0.95 and
                self.tcn_score_corr > 0.95):
            lines.append("  ✅ INT8 accuracy is excellent — safe to deploy.")
        elif (self.det_mean_iou > 0.80 and self.tcn_top1_agree > 0.90 and
              self.tcn_score_corr > 0.90):
            lines.append("  ⚠️  Moderate accuracy drop — acceptable for most use-cases.")
        else:
            lines.append("  ❌ Significant accuracy loss — review calibration data,")
            lines.append("     increase --max-samples, or switch to FP16 RKNN.")
        return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RKNN INT8 accuracy vs PyTorch FP32 baseline"
    )
    parser.add_argument("--video", required=True, help="Test video for comparison")
    parser.add_argument(
        "--detector-pt", default=None,
        help="PyTorch YOLO checkpoint (e.g. checkpoint/yolo26m.pt) for FP32 baseline",
    )
    parser.add_argument(
        "--detector-rknn", default=None,
        help="RKNN YOLO model (.rknn) for INT8 candidate",
    )
    parser.add_argument(
        "--temporal-pt", default=None,
        help="PyTorch TCN checkpoint for FP32 baseline",
    )
    parser.add_argument(
        "--temporal-rknn", default=None,
        help="RKNN TCN model (.rknn) for INT8 candidate",
    )
    parser.add_argument(
        "--temporal-labels", default=None,
        help="JSON list of class labels (if using --temporal-rknn without checkpoint)",
    )
    parser.add_argument("--output-dir", default="runs/rknn_validation")
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max frames to compare (speed up validation)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Imports ──────────────────────────────────────────────────────────
    from edgefall.infer.pipeline import EdgeFallPipeline
    from edgefall.infer.state_machine import AlarmConfig
    from edgefall.infer.tracker import IoUTracker
    from edgefall.infer_video import run_fall_detection_on_video
    from edgefall.utils.deps import require_torch

    torch = require_torch()

    # ── Labels ───────────────────────────────────────────────────────────
    if args.temporal_labels:
        with open(args.temporal_labels, "r") as fh:
            labels = json.load(fh)
    elif args.temporal_pt:
        ckpt = torch.load(args.temporal_pt, map_location="cpu", weights_only=False)
        labels = ckpt["labels"]
    else:
        labels = ["walk","fall","fallen","sit_down","sitting","lie_down","lying",
                   "stand_up","standing","other","kneel_down","kneeling",
                   "squat_down","squatting","crawl","jump"]

    # ── Run both pipelines and collect frame-level data ──────────────────

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    print(f"Video: {args.video} ({w}x{h}, {fps:.1f} fps, {total} frames)")
    print(f"Comparing up to {args.max_frames} frames...\n")

    # --- PT baseline ---
    print("[1/4] Running PyTorch FP32 baseline ...")
    from edgefall.models.detector import UltralyticsPersonDetector
    from edgefall.models.temporal_tcn import build_geometry_tcn

    if args.detector_pt and args.temporal_pt:
        pt_detector = UltralyticsPersonDetector(
            weights=args.detector_pt, conf=args.conf, imgsz=args.imgsz, device="cpu")
        ckpt = torch.load(args.temporal_pt, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        pt_tcn = build_geometry_tcn(
            geometry_dim=int(cfg["data"]["geometry_dim"]),
            num_classes=len(labels),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            num_layers=int(cfg["model"]["num_layers"]),
            dropout=float(cfg["model"]["dropout"]),
        )
        pt_tcn.load_state_dict(ckpt["model"])
        pt_tcn.eval()

        pt_results = _run_pipeline_frames(
            args.video, pt_detector, pt_tcn, labels, fps, args.max_frames, args.conf,
        )
    else:
        print("  Skipped (no PT paths provided)")
        pt_results = None

    # --- RKNN candidate ---
    print("[2/4] Running RKNN INT8 candidate ...")
    if args.detector_rknn and args.temporal_rknn:
        from edgefall.models.detector_rknn import RKNNDetector
        from edgefall.models.temporal_tcn_rknn import RKNNTemporalHead

        rknn_detector = RKNNDetector(
            rknn_path=args.detector_rknn, conf=args.conf, imgsz=args.imgsz)
        rknn_tcn = RKNNTemporalHead(args.temporal_rknn)

        rknn_results = _run_pipeline_frames(
            args.video, rknn_detector, rknn_tcn, labels, fps, args.max_frames, args.conf,
        )
        rknn_detector.release()
        rknn_tcn.release()
    else:
        print("  Skipped (no RKNN paths provided)")
        rknn_results = None

    if pt_results is None or rknn_results is None:
        print("Need both PT and RKNN paths to compare. Exiting.")
        return

    # ── Compare ──────────────────────────────────────────────────────────
    print("[3/4] Computing accuracy metrics ...")
    report = _compute_report(pt_results, rknn_results)

    # ── Save ─────────────────────────────────────────────────────────────
    print("[4/4] Saving report ...")
    report_path = out_dir / "accuracy_report.txt"
    report_path.write_text(report.summary(), encoding="utf-8")

    metrics_json = out_dir / "accuracy_metrics.json"
    metrics_json.write_text(json.dumps({
        "det_mean_iou": report.det_mean_iou,
        "det_count_delta_pct": report.det_count_delta_pct,
        "det_box_rmse": report.det_box_rmse,
        "tcn_logit_cosine": report.tcn_logit_cosine,
        "tcn_top1_agree": report.tcn_top1_agree,
        "tcn_score_corr": report.tcn_score_corr,
        "e2e_alarm_iou": report.e2e_alarm_iou,
        "e2e_alarm_recall": report.e2e_alarm_recall,
        "frames_compared": report.frames_compared,
    }, indent=2), encoding="utf-8")

    print("\n" + report.summary())
    print(f"\nReport saved to {report_path}")
    print(f"Metrics JSON: {metrics_json}")


# ── Per-frame pipeline runner ────────────────────────────────────────────


def _run_pipeline_frames(video_path, detector, tcn, labels, fps, max_frames, conf):
    """Run pipeline frame-by-frame and collect detector+TCN outputs."""
    from edgefall.data.geometry import Box, geometry_sequence
    from edgefall.infer.tracker import IoUTracker
    from edgefall.infer.state_machine import AlarmConfig, FallStateMachine
    from collections import defaultdict, deque
    import torch as _t

    cap = cv2.VideoCapture(video_path)
    tracker = IoUTracker(iou_threshold=0.1, max_age=75, min_hits=1)
    histories = defaultdict(lambda: deque(maxlen=16))
    machines = defaultdict(lambda: FallStateMachine(AlarmConfig(
        min_fall_transition_frames=1,
        allow_fall_transition_alarm=True,
        min_fall_transition_alarm_frames=1,
        fall_transition_score_threshold=0.0,
        min_fallen_seconds=0.02,
        allow_fallen_without_transition=True,
        min_fallen_without_transition_seconds=0.02,
        fallen_score_threshold=0.0,
        fps=fps,
        descent_threshold=0.0,
        aspect_change_threshold=0.0,
        fall_transition_label="fall",
    )))
    positive_indices = [i for i, lbl in enumerate(labels) if lbl in ("fall", "fallen")]

    results = []  # list of per-frame dicts
    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        detections = detector.predict(frame)
        tracks = tracker.update(detections)

        frame_data = {"frame": frame_idx, "detections": [], "tcn_outputs": [],
                      "alarms": []}

        # Record raw detections for IoU comparison
        for det in detections:
            frame_data["detections"].append(
                [det.x1, det.y1, det.x2, det.y2, det.score])

        for track in tracks:
            histories[track.track_id].append(
                Box(track.box.x1, track.box.y1, track.box.x2, track.box.y2))
            if len(histories[track.track_id]) < 16:
                continue
            history = list(histories[track.track_id])
            features = geometry_sequence(history[-16:], w, h)
            x = _t.tensor([features], dtype=_t.float32)
            with _t.no_grad():
                logits = tcn(x)
                if hasattr(logits, 'detach'):
                    logits = logits.detach().cpu().numpy()
                probs = _t.softmax(_t.from_numpy(logits) if not isinstance(logits, _t.Tensor) else logits, dim=-1)
                if hasattr(probs, 'numpy'):
                    probs = probs.numpy()
            class_idx = int(np.argmax(probs[0]))
            positive_score = float(probs[0][positive_indices].max()) if positive_indices else float(probs[0][class_idx])

            frame_data["tcn_outputs"].append({
                "track_id": track.track_id,
                "logits": logits[0].tolist(),
                "class_idx": class_idx,
                "positive_score": positive_score,
            })

            label = labels[class_idx]
            delta_y = features[-1][5]
            delta_aspect = features[-1][6]
            event = machines[track.track_id].update(
                track.track_id, label, positive_score, delta_y, delta_aspect)
            if event.alarm:
                frame_data["alarms"].append({
                    "track_id": track.track_id,
                    "frame": frame_idx,
                    "score": positive_score,
                })

        results.append(frame_data)
        frame_idx += 1

    cap.release()
    return results


# ── Metric computation ────────────────────────────────────────────────────


def _compute_report(pt_results, rknn_results):
    report = ValidationReport()

    n = min(len(pt_results), len(rknn_results))
    report.frames_compared = n

    # --- Detector metrics ---
    ious = []
    count_deltas = []
    box_rmses = []

    for i in range(n):
        pt_dets = pt_results[i]["detections"]
        rk_dets = rknn_results[i]["detections"]

        count_deltas.append(
            (len(rk_dets) - len(pt_dets)) / max(len(pt_dets), 1) * 100)

        # Greedy IoU matching
        matched = set()
        for pt_box in pt_dets:
            best_iou = 0.0
            best_j = -1
            for j, rk_box in enumerate(rk_dets):
                if j in matched:
                    continue
                iou_val = _iou(pt_box, rk_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0:
                matched.add(best_j)
                ious.append(best_iou)
                # Box RMSE (on matched pairs)
                pt_arr = np.array(pt_box[:4])
                rk_arr = np.array(rk_dets[best_j][:4])
                box_rmses.append(float(np.sqrt(np.mean((pt_arr - rk_arr) ** 2))))

    report.det_mean_iou = float(np.mean(ious)) if ious else 0.0
    report.det_count_delta_pct = float(np.mean(count_deltas)) if count_deltas else 0.0
    report.det_box_rmse = float(np.mean(box_rmses)) if box_rmses else 0.0

    # --- TCN metrics ---
    cosines = []
    top1_agrees = []
    score_deltas = []

    for i in range(n):
        pt_tcns = {t["track_id"]: t for t in pt_results[i]["tcn_outputs"]}
        rk_tcns = {t["track_id"]: t for t in rknn_results[i]["tcn_outputs"]}
        for tid in pt_tcns:
            if tid not in rk_tcns:
                continue
            pt_logits = np.array(pt_tcns[tid]["logits"])
            rk_logits = np.array(rk_tcns[tid]["logits"])
            cosines.append(_cosine_sim(pt_logits, rk_logits))
            top1_agrees.append(
                1.0 if pt_tcns[tid]["class_idx"] == rk_tcns[tid]["class_idx"] else 0.0)
            score_deltas.append(
                abs(pt_tcns[tid]["positive_score"] - rk_tcns[tid]["positive_score"]))

    report.tcn_logit_cosine = float(np.mean(cosines)) if cosines else 0.0
    report.tcn_top1_agree = float(np.mean(top1_agrees)) if top1_agrees else 0.0

    # Pearson correlation of positive scores
    pt_scores, rk_scores = [], []
    for i in range(n):
        pt_tcns = {t["track_id"]: t for t in pt_results[i]["tcn_outputs"]}
        rk_tcns = {t["track_id"]: t for t in rknn_results[i]["tcn_outputs"]}
        for tid in pt_tcns:
            if tid in rk_tcns:
                pt_scores.append(pt_tcns[tid]["positive_score"])
                rk_scores.append(rk_tcns[tid]["positive_score"])
    if len(pt_scores) > 2:
        pt_arr, rk_arr = np.array(pt_scores), np.array(rk_scores)
        report.tcn_score_corr = float(
            np.corrcoef(pt_arr, rk_arr)[0, 1])
    else:
        report.tcn_score_corr = 1.0

    # --- End-to-end alarm ---
    pt_alarms = [(e["frame"], e["track_id"]) for r in pt_results for e in r["alarms"]]
    rk_alarms = set((e["frame"], e["track_id"]) for r in rknn_results for e in r["alarms"])
    matched_alarms = sum(1 for a in pt_alarms if a in rk_alarms)
    report.e2e_alarm_recall = matched_alarms / max(len(pt_alarms), 1)
    report.e2e_alarm_iou = matched_alarms / max(len(set(pt_alarms) | rk_alarms), 1)

    return report


if __name__ == "__main__":
    main()
