#!/usr/bin/env python3
"""Search alarm post-processing parameters for Recall@FP/video metrics."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from eval_alarm_metrics import (
    average_precision,
    load_events,
    load_videos,
    match_events,
    precision_at_recall,
    recall_at_fp_per_video,
    topk_per_video,
)
from search_event_postprocess import parse_float_list, postprocess, write_events


def f1_at_k(events: list[dict], ground_truth: list[dict], k: int, iou_threshold: float) -> float:
    selected = topk_per_video(events, k)
    tp, fp, fn = match_events(selected, ground_truth, iou_threshold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--fp-per-video", type=float, default=3.0)
    parser.add_argument("--pad-before-values", default="0,0.5,1,1.5,2")
    parser.add_argument("--pad-after-values", default="0,0.5,1,1.5,2")
    parser.add_argument("--merge-gap-values", default="0,0.5,1,2,3")
    parser.add_argument("--score-modes", default="max,duration_penalty,sqrt_duration_penalty,duration_divide")
    parser.add_argument("--duration-weight-values", default="0,0.005,0.01,0.02,0.04")
    parser.add_argument("--min-raw-score-values", default="0")
    parser.add_argument("--min-duration-values", default="0")
    parser.add_argument("--max-duration-values", default="1000000000")
    parser.add_argument("--min-score-values", default="-1000000000")
    parser.add_argument("--select-per-video-values", default="none")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    raw_events = load_events(args.input)
    ground_truth = load_events(args.ground_truth)
    videos = load_videos(args.manifest, raw_events, ground_truth)
    modes = [part.strip() for part in args.score_modes.split(",") if part.strip()]
    select_modes = [part.strip() for part in args.select_per_video_values.split(",") if part.strip()]

    best = None
    rows = []
    for (
        pad_before,
        pad_after,
        merge_gap,
        mode,
        duration_weight,
        min_raw_score,
        min_duration,
        max_duration,
        min_score,
        select_per_video,
    ) in itertools.product(
        parse_float_list(args.pad_before_values),
        parse_float_list(args.pad_after_values),
        parse_float_list(args.merge_gap_values),
        modes,
        parse_float_list(args.duration_weight_values),
        parse_float_list(args.min_raw_score_values),
        parse_float_list(args.min_duration_values),
        parse_float_list(args.max_duration_values),
        parse_float_list(args.min_score_values),
        select_modes,
    ):
        events = postprocess(
            raw_events,
            pad_before,
            pad_after,
            merge_gap,
            mode,
            duration_weight,
            min_raw_score,
            min_duration,
            max_duration,
            min_score,
            select_per_video,
        )
        recall_budget = recall_at_fp_per_video(
            events,
            ground_truth,
            len(videos),
            args.fp_per_video,
            args.iou_threshold,
        )
        f1_5 = f1_at_k(events, ground_truth, 5, args.iou_threshold)
        f1_10 = f1_at_k(events, ground_truth, 10, args.iou_threshold)
        ap50 = average_precision(topk_per_video(events, 50), ground_truth, args.iou_threshold)
        p90 = precision_at_recall(events, ground_truth, 0.90, args.iou_threshold)
        p95 = precision_at_recall(events, ground_truth, 0.95, args.iou_threshold)
        matched = match_events(events, ground_truth, args.iou_threshold)[0]
        row = {
            "recall_at_3_fp_per_video": recall_budget["recall"],
            "tp_at_3_fp_per_video": recall_budget["tp"],
            "fp_at_3_fp_per_video": recall_budget["fp"],
            "actual_fp_per_video": recall_budget["fp_per_video"],
            "f1_at_5_video": f1_5,
            "f1_at_10_video": f1_10,
            "ap_at_top50_video": ap50,
            "p90": p90,
            "p95": p95,
            "matched": matched,
            "events": len(events),
            "pad_before": pad_before,
            "pad_after": pad_after,
            "merge_gap": merge_gap,
            "score_mode": mode,
            "duration_weight": duration_weight,
            "min_raw_score": min_raw_score,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "min_score": min_score,
            "select_per_video": select_per_video,
        }
        rows.append(row)
        key = (
            row["recall_at_3_fp_per_video"],
            row["f1_at_5_video"],
            row["f1_at_10_video"],
            row["ap_at_top50_video"],
            row["p95"],
            row["p90"],
            row["matched"],
            -row["events"],
        )
        if best is None or key > best[0]:
            best = (key, row, events)

    assert best is not None
    output = Path(args.output)
    write_events(output, best[2])
    rows.sort(
        key=lambda row: (
            row["recall_at_3_fp_per_video"],
            row["f1_at_5_video"],
            row["f1_at_10_video"],
            row["ap_at_top50_video"],
            row["p95"],
            row["p90"],
            row["matched"],
            -row["events"],
        ),
        reverse=True,
    )
    print(json.dumps({"best": best[1], "output": str(output)}, indent=2, sort_keys=True))
    print("top candidates:")
    for row in rows[: args.top_k]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
