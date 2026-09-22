#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EdgeFallDetection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int class_id;
} EdgeFallDetection;

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
    int max_detections);

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
    int max_detections);

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
    int max_detections);

#ifdef __cplusplus
}
#endif
