#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

struct Candidate {
  int anchor;
  int cls;
  float score;
};

struct ModelZooCandidate {
  float score;
  int cls;
  float x1;
  float y1;
  float x2;
  float y2;
};

float iou(const EdgeFallDetection& a, const EdgeFallDetection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter;
  return denom > 0.0f ? inter / denom : 0.0f;
}

void partial_descending(std::vector<int>& idx, const std::vector<float>& scores, int k) {
  if (k < static_cast<int>(idx.size())) {
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
      return scores[a] > scores[b];
    });
    idx.resize(k);
  }
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return scores[a] > scores[b];
  });
}

void compute_dfl(const float* logits, int reg_max, float* out) {
  for (int side = 0; side < 4; ++side) {
    const float* side_logits = logits + side * reg_max;
    float max_value = side_logits[0];
    for (int i = 1; i < reg_max; ++i) {
      max_value = std::max(max_value, side_logits[i]);
    }
    float denom = 0.0f;
    float weighted = 0.0f;
    for (int i = 0; i < reg_max; ++i) {
      const float value = std::exp(side_logits[i] - max_value);
      denom += value;
      weighted += value * static_cast<float>(i);
    }
    out[side] = denom > 0.0f ? weighted / denom : 0.0f;
  }
}

void process_modelzoo_branch(
    const float* box,
    const float* cls,
    const float* score_sum,
    int h,
    int w,
    int num_classes,
    int reg_max,
    int box_channels,
    float stride,
    float conf_threshold,
    int person_class_id,
    std::vector<ModelZooCandidate>& candidates) {
  if (!box || !cls || !score_sum || h <= 0 || w <= 0 || num_classes <= 0 ||
      reg_max <= 0 || box_channels <= 0) {
    return;
  }
  if (box_channels != 4 && box_channels != 4 * reg_max) {
    return;
  }

  const int spatial = h * w;
  float dfl_logits[128];
  float dist[4];

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int offset = y * w + x;
      if (score_sum[offset] < conf_threshold) {
        continue;
      }

      int cls_id = 0;
      float best_score = cls[offset];
      for (int c = 1; c < num_classes; ++c) {
        const float score = cls[c * spatial + offset];
        if (score > best_score) {
          best_score = score;
          cls_id = c;
        }
      }
      if (cls_id != person_class_id || best_score < conf_threshold) {
        continue;
      }

      if (box_channels == 4) {
        for (int c = 0; c < 4; ++c) {
          dist[c] = box[c * spatial + offset];
        }
      } else {
        for (int c = 0; c < box_channels; ++c) {
          dfl_logits[c] = box[c * spatial + offset];
        }
        compute_dfl(dfl_logits, reg_max, dist);
      }

      const float cx = static_cast<float>(x) + 0.5f;
      const float cy = static_cast<float>(y) + 0.5f;
      candidates.push_back({
          best_score,
          cls_id,
          (cx - dist[0]) * stride,
          (cy - dist[1]) * stride,
          (cx + dist[2]) * stride,
          (cy + dist[3]) * stride,
      });
    }
  }
}

}  // namespace

int edgefall_yolo_postprocess(
    const float* boxes,
    const float* class_scores,
    int num_anchors,
    int num_classes,
    float conf_threshold,
    int person_class_id,
    int anchor_topk,
    int final_topk,
    float nms_iou_threshold,
    float scale,
    float pad_left,
    float pad_top,
    EdgeFallDetection* out_detections,
    int max_detections) {
  if (!boxes || !class_scores || !out_detections || num_anchors <= 0 ||
      num_classes <= 0 || max_detections <= 0 || scale <= 0.0f) {
    return 0;
  }

  if (anchor_topk <= 0 || anchor_topk > num_anchors) {
    anchor_topk = num_anchors;
  }
  if (final_topk <= 0) {
    final_topk = 300;
  }

  std::vector<float> anchor_scores(num_anchors, 0.0f);
  for (int a = 0; a < num_anchors; ++a) {
    float best = class_scores[a * num_classes];
    for (int c = 1; c < num_classes; ++c) {
      best = std::max(best, class_scores[a * num_classes + c]);
    }
    anchor_scores[a] = best;
  }

  std::vector<int> anchor_idx(num_anchors);
  std::iota(anchor_idx.begin(), anchor_idx.end(), 0);
  partial_descending(anchor_idx, anchor_scores, anchor_topk);

  std::vector<Candidate> candidates;
  candidates.reserve(static_cast<size_t>(anchor_idx.size()) * static_cast<size_t>(num_classes));
  for (int anchor : anchor_idx) {
    const float* score_row = class_scores + static_cast<size_t>(anchor) * num_classes;
    for (int cls = 0; cls < num_classes; ++cls) {
      candidates.push_back({anchor, cls, score_row[cls]});
    }
  }

  if (final_topk < static_cast<int>(candidates.size())) {
    std::nth_element(candidates.begin(), candidates.begin() + final_topk, candidates.end(),
                     [](const Candidate& a, const Candidate& b) {
                       return a.score > b.score;
                     });
    candidates.resize(final_topk);
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.score > b.score;
  });

  std::vector<EdgeFallDetection> filtered;
  filtered.reserve(candidates.size());
  for (const Candidate& cand : candidates) {
    if (cand.score < conf_threshold || cand.cls != person_class_id) {
      continue;
    }
    const float* b = boxes + static_cast<size_t>(cand.anchor) * 4;
    EdgeFallDetection det;
    det.x1 = std::max(0.0f, (b[0] - pad_left) / scale);
    det.y1 = std::max(0.0f, (b[1] - pad_top) / scale);
    det.x2 = (b[2] - pad_left) / scale;
    det.y2 = (b[3] - pad_top) / scale;
    det.score = cand.score;
    det.class_id = cand.cls;
    filtered.push_back(det);
  }

  std::vector<EdgeFallDetection> kept;
  std::vector<char> suppressed(filtered.size(), 0);
  for (size_t i = 0; i < filtered.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(filtered[i]);
    if (static_cast<int>(kept.size()) >= max_detections) {
      break;
    }
    for (size_t j = i + 1; j < filtered.size(); ++j) {
      if (!suppressed[j] && iou(filtered[i], filtered[j]) > nms_iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }

  const int count = std::min(static_cast<int>(kept.size()), max_detections);
  for (int i = 0; i < count; ++i) {
    out_detections[i] = kept[i];
  }
  return count;
}

int edgefall_yolo_postprocess_modelzoo(
    const float* box_s8,
    const float* cls_s8,
    const float* score_sum_s8,
    int h_s8,
    int w_s8,
    const float* box_s16,
    const float* cls_s16,
    const float* score_sum_s16,
    int h_s16,
    int w_s16,
    const float* box_s32,
    const float* cls_s32,
    const float* score_sum_s32,
    int h_s32,
    int w_s32,
    int model_size,
    int num_classes,
    int reg_max,
    float conf_threshold,
    int person_class_id,
    int final_topk,
    float nms_iou_threshold,
    float scale,
    float pad_left,
    float pad_top,
    EdgeFallDetection* out_detections,
    int max_detections) {
  return edgefall_yolo_postprocess_modelzoo_v2(
      box_s8, cls_s8, score_sum_s8, h_s8, w_s8,
      box_s16, cls_s16, score_sum_s16, h_s16, w_s16,
      box_s32, cls_s32, score_sum_s32, h_s32, w_s32,
      model_size, num_classes, reg_max, 4 * reg_max,
      conf_threshold, person_class_id, final_topk, nms_iou_threshold,
      scale, pad_left, pad_top, out_detections, max_detections);
}

int edgefall_yolo_postprocess_modelzoo_v2(
    const float* box_s8,
    const float* cls_s8,
    const float* score_sum_s8,
    int h_s8,
    int w_s8,
    const float* box_s16,
    const float* cls_s16,
    const float* score_sum_s16,
    int h_s16,
    int w_s16,
    const float* box_s32,
    const float* cls_s32,
    const float* score_sum_s32,
    int h_s32,
    int w_s32,
    int model_size,
    int num_classes,
    int reg_max,
    int box_channels,
    float conf_threshold,
    int person_class_id,
    int final_topk,
    float nms_iou_threshold,
    float scale,
    float pad_left,
    float pad_top,
    EdgeFallDetection* out_detections,
    int max_detections) {
  if (!out_detections || model_size <= 0 || num_classes <= 0 || reg_max <= 0 ||
      max_detections <= 0 || scale <= 0.0f) {
    return 0;
  }
  if (reg_max > 32) {
    return 0;
  }
  if (box_channels != 4 && box_channels != 4 * reg_max) {
    return 0;
  }
  if (final_topk <= 0) {
    final_topk = 300;
  }

  std::vector<ModelZooCandidate> candidates;
  candidates.reserve(1024);
  process_modelzoo_branch(box_s8, cls_s8, score_sum_s8, h_s8, w_s8, num_classes, reg_max, box_channels,
                          h_s8 > 0 ? static_cast<float>(model_size) / static_cast<float>(h_s8) : 8.0f,
                          conf_threshold, person_class_id, candidates);
  process_modelzoo_branch(box_s16, cls_s16, score_sum_s16, h_s16, w_s16, num_classes, reg_max, box_channels,
                          h_s16 > 0 ? static_cast<float>(model_size) / static_cast<float>(h_s16) : 16.0f,
                          conf_threshold, person_class_id, candidates);
  process_modelzoo_branch(box_s32, cls_s32, score_sum_s32, h_s32, w_s32, num_classes, reg_max, box_channels,
                          h_s32 > 0 ? static_cast<float>(model_size) / static_cast<float>(h_s32) : 32.0f,
                          conf_threshold, person_class_id, candidates);

  if (final_topk < static_cast<int>(candidates.size())) {
    std::nth_element(candidates.begin(), candidates.begin() + final_topk, candidates.end(),
                     [](const ModelZooCandidate& a, const ModelZooCandidate& b) {
                       return a.score > b.score;
                     });
    candidates.resize(final_topk);
  }
  std::sort(candidates.begin(), candidates.end(), [](const ModelZooCandidate& a, const ModelZooCandidate& b) {
    return a.score > b.score;
  });

  std::vector<EdgeFallDetection> filtered;
  filtered.reserve(candidates.size());
  for (const ModelZooCandidate& cand : candidates) {
    EdgeFallDetection det;
    det.x1 = std::max(0.0f, (cand.x1 - pad_left) / scale);
    det.y1 = std::max(0.0f, (cand.y1 - pad_top) / scale);
    det.x2 = (cand.x2 - pad_left) / scale;
    det.y2 = (cand.y2 - pad_top) / scale;
    det.score = cand.score;
    det.class_id = cand.cls;
    filtered.push_back(det);
  }

  std::vector<EdgeFallDetection> kept;
  std::vector<char> suppressed(filtered.size(), 0);
  for (size_t i = 0; i < filtered.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(filtered[i]);
    if (static_cast<int>(kept.size()) >= max_detections) {
      break;
    }
    for (size_t j = i + 1; j < filtered.size(); ++j) {
      if (!suppressed[j] && iou(filtered[i], filtered[j]) > nms_iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }

  const int count = std::min(static_cast<int>(kept.size()), max_detections);
  for (int i = 0; i < count; ++i) {
    out_detections[i] = kept[i];
  }
  return count;
}
