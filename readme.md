# EdgeFall RK3588 部署说明：YOLO26n No-Attn Person 480 + Best Params

本部署包适用于瑞芯微RK3588，已经包含 RKNN 模型、板端推理脚本、src/edgefall 运行代码和 deploy_rk3588 C++ 后处理库。

完整的云端训练代码请见https://github.com/BaiHJ-201/edge-fall.git。

本部署包使用当前推荐的 RK3588 推理组合：

```text
检测器：YOLO26n no-attn person-only 480，model_zoo 9 输出 INT8 RKNN
TCN： mixed direct YOLO26n GeometryTCN，默认使用 FP16 RKNN.
```

## 1. 模型文件

默认推荐 detector：

```text
runs/final/export/yolo26n_noattn_person_int8_imgsz480_modelzoo_a043ir_450_50_rk3588.rknn
```

默认推荐 TCN模型：

```text
runs/final/export/temporal_head_yolo26n_noattn_person_640_fp16_ln_addrelu_decomposed_cliprelu_rk3588.rknn
```

## 2. 设置板端环境

在 RK3588 板端进入部署包目录后设置环境变量：

```bash
export PYTHONPATH=src
export EDGEFALL_YOLO_POSTPROCESS_SO=deploy_rk3588/libedgefall_yolo_postprocess.so
```

复制部署包到板端后，先重新编译 YOLO C++ 后处理库：

```bash
cd ~/deploy_rk3588
make clean all
nm -D libedgefall_yolo_postprocess.so | grep edgefall_yolo_postprocess_modelzoo_v2
```

## 3. 单视频可视化推理

日常板端演示或单视频部署，只需要运行下面这一条命令。注：examples/01.mp4是要推理的视频
```bash
cd ..
PYTHONPATH=src python3 scripts/infer_video_rknn_visual.py \
  --video examples/01.mp4 \
  --detector-rknn runs/final/export/yolo26n_noattn_person_int8_imgsz480_modelzoo_a043ir_450_50_rk3588.rknn \
  --temporal-rknn runs/final/export/temporal_head_yolo26n_noattn_person_640_fp16_ln_addrelu_decomposed_cliprelu_rk3588.rknn \
  --temporal-labels runs/final/export/temporal_labels_yolo26n_noattn_person_640.json \
  --output-video examples/fall_annotated_rknn_480_best.mp4 \
  --output-events examples/events_rknn_480_best.jsonl \
  --conf 0.3 \
  --imgsz 480 \
  --person-class-id 0 \
  --num-classes 1 \
  --tracker-iou-threshold 0.05 \
  --tracker-min-hits 2 \
  --tracker-max-age 30 \
  --min-fall-transition-frames 1 \
  --allow-fall-transition-alarm \
  --fall-transition-alarm-frames 1 \
  --fall-transition-score-threshold 0.0 \
  --min-fallen-seconds 0.02 \
  --allow-fallen-without-transition \
  --fallen-without-transition-seconds 0.02 \
  --fallen-score-threshold 0.0 \
  --descent-threshold 0.0 \
  --aspect-change-threshold 0.0 \
  --alarm-persist-frames 120
```

这条命令会同时输出：

```text
examples/fall_annotated_rknn_480_best.mp4
examples/events_rknn_480_best.jsonl
```

单视频演示或普通可视化检查不需要再跑离线 alarm postprocess。

### 3.1 红外视频推理

红外视频可以直接作为 `--video` 输入。样例 A043 AVI 由 OpenCV 解码后是 3 通道等值帧；如果板端红外相机或其他视频源输出单通道灰度帧，当前 `RKNNDetector` 会自动复制为 3 通道后再做 480 letterbox、BGR->RGB、NCHW 和 `/255` 归一化。

```bash
PYTHONPATH=src python3 scripts/infer_video_rknn_visual.py \
  --video /path/to/A043_or_ir_video.avi \
  --detector-rknn runs/final/export/yolo26n_noattn_person_int8_imgsz480_modelzoo_a043ir_450_50_rk3588.rknn \
  --temporal-rknn runs/final/export/temporal_head_yolo26n_noattn_person_640_fp16_ln_addrelu_decomposed_cliprelu_rk3588.rknn \
  --temporal-labels runs/final/export/temporal_labels_yolo26n_noattn_person_640.json \
  --output-video out/ir_annotated_rknn_480_best.mp4 \
  --output-events out/ir_events_rknn_480_best.jsonl \
  --conf 0.03 \
  --imgsz 480 \
  --person-class-id 0 \
  --num-classes 1 \
  --tracker-iou-threshold 0.05 \
  --tracker-min-hits 2 \
  --tracker-max-age 30 \
  --min-fall-transition-frames 1 \
  --allow-fall-transition-alarm \
  --fall-transition-alarm-frames 1 \
  --fall-transition-score-threshold 0.0 \
  --min-fallen-seconds 0.02 \
  --allow-fallen-without-transition \
  --fallen-without-transition-seconds 0.02 \
  --fallen-score-threshold 0.0 \
  --descent-threshold 0.0 \
  --aspect-change-threshold 0.0 \
  --alarm-persist-frames 120
```

如果只需要 JSONL 事件输出，不需要视频标注结果，可以运行：

```bash
PYTHONPATH=src python3 scripts/infer_video_rknn.py \
  --video /path/to/test_video.mp4 \
  --detector-rknn runs/final/export/yolo26n_noattn_person_int8_imgsz480_modelzoo_a043ir_450_50_rk3588.rknn \
  --temporal-rknn runs/final/export/temporal_head_yolo26n_noattn_person_640_fp16_ln_addrelu_decomposed_cliprelu_rk3588.rknn \
  --temporal-labels runs/final/export/temporal_labels_yolo26n_noattn_person_640.json \
  --output out/events_rknn_480_best.jsonl \
  --conf 0.02 \
  --imgsz 480 \
  --person-class-id 0 \
  --num-classes 1 \
  --tracker-iou-threshold 0.05 \
  --tracker-min-hits 2 \
  --tracker-max-age 30 \
  --min-fall-transition-frames 1 \
  --allow-fall-transition-alarm \
  --fall-transition-alarm-frames 1 \
  --fall-transition-score-threshold 0.0 \
  --min-fallen-seconds 0.02 \
  --allow-fallen-without-transition \
  --fallen-without-transition-seconds 0.02 \
  --fallen-score-threshold 0.0 \
  --descent-threshold 0.0 \
  --aspect-change-threshold 0.0
```

## 4. 数据集推理

这一节不是单一视频部署流程。只有在板端使用数据集评估p90和p95指标时才需要执行。

该流程需要：

```text
manifest: 按照manifest格式准备的数据集
ground-truth events
raw events
```

其中 `infer_manifest_events_rknn.py` 负责按 manifest格式 跑 RKNN 推理并输出 raw events；`search_alarm_postprocess.py` 负责按 ground truth 做 alarm 后处理和指标搜索。`merge_gap_frames=75` 只在 manifest 复测里使用。

RKNN manifest 事件推理：

```bash
PYTHONPATH=src python3 scripts/infer_manifest_events_rknn.py \
  --manifest /path/to/manifest.jsonl \
  --detector-rknn runs/final/export/yolo26n_noattn_person_int8_imgsz480_modelzoo_a043ir_450_50_rk3588.rknn \
  --temporal-rknn runs/final/export/temporal_head_yolo26n_noattn_person_640_fp16_ln_addrelu_decomposed_cliprelu_rk3588.rknn \
  --temporal-labels runs/final/export/temporal_labels_yolo26n_noattn_person_640.json \
  --output out/events_rknn_480_best_raw.jsonl \
  --conf 0.02 \
  --imgsz 480 \
  --person-class-id 0 \
  --num-classes 1 \
  --tracker-iou-threshold 0.05 \
  --tracker-min-hits 2 \
  --tracker-max-age 30 \
  --min-fall-transition-frames 1 \
  --allow-fall-transition-alarm \
  --fall-transition-alarm-frames 1 \
  --fall-transition-score-threshold 0.0 \
  --min-fallen-seconds 0.02 \
  --allow-fallen-without-transition \
  --fallen-without-transition-seconds 0.02 \
  --fallen-score-threshold 0.0 \
  --descent-threshold 0.0 \
  --aspect-change-threshold 0.0 \
  --merge-gap-frames 75
```

然后运行alarm postprocess。

```bash
PYTHONPATH=src python3 scripts/search_alarm_postprocess.py \
  --input out/events_rknn_480_best_raw.jsonl \
  --ground-truth /path/to/ground_truth_events.jsonl \
  --manifest /path/to/manifest.jsonl \
  --output out/events_rknn_480_best_post.jsonl \
  --pad-before-values 0.5 \
  --pad-after-values 0.5 \
  --merge-gap-values 3.0 \
  --score-modes max \
  --duration-weight-values 0.0 \
  --min-raw-score-values 0.0 \
  --min-duration-values 0.0 \
  --select-per-video-values none \
  --top-k 10
```

其他推荐参数：Alarm postprocess 参数，跨场景/跨视角：

```text
pad_before=0.5
pad_after=0.0
merge_gap=0.0
score_mode=duration_divide
duration_weight=0.0
min_duration=0.0
min_raw_score=0.0
select_per_video=none
```

## 5. 模型单视频指标测量结果✅

测量日期：2026-07-02，板端 RK3588，librknnrt 1.6.0 / driver 0.9.8 / rknn-toolkit-lite2 2.3.2。
NPU 单核运行（模型编译时 `single_core_mode=True`）。输入 480×480。

### 5.1 模型参数量（≤ 20M ✅）

| 模块 | 参数量 | 说明 |
|------|--------|------|
| YOLO Detector (YOLOv6-N no-attn person-only) | ~3,100,000 | |
| TCN (GeometryTCN) | 51,280 |  |
| **合计** | **~3,151,280** | **≤ 20M ✓** |

### 5.2 总参数量 FP32 大小（≤ 80 MB ✅）

```
3,151,280 params × 4 bytes/param = 12,605,120 bytes ≈ 12.0 MB
```

**≤ 80 MB ✓**

### 5.3 模型文件大小（480）

| 模型 | 大小 | 精度 |
|------|------|------|
| YOLO Detector (480 INT8) | 3,520.1 KB (3,690,375 B) | INT8 |
| TCN | 432.8 KB (443,191 B) | FP16 |

### 5.4 端侧推理耗时 — 480 INT8 + TCN FP16 ✅

测量方法：50 次推理取平均，warmup 10 次。输入 480×480，单核 NPU。预处理输入 1280×720→480。

| 阶段 | 耗时 | 占比 |
|------|------|------|
| ① 预处理 (resize 1280×720→480 / BGR→RGB / NCHW) | 3.3 ms | 6.0% |
| ② NPU 推理 (YOLO backbone+head) | 48.8 ms | 88.7% |
| ③ C++ 后处理 (DFL/NMS/坐标, modelzoo_v2) | ~2.0 ms | 3.6% |
| ④ TCN per track | 0.5 ms | — |
| ⑤ TCN × 3 tracks | 1.5 ms | 2.7% |
| **端到端合计（含 TCN×3）** | **~55.6 ms** | |

> 480 INT8 端到端延时 ~55.6ms，远低于 100ms 阈值，相比 640 INT8（101.7ms）加速约 **45%**。NPU 推理从 89.5ms 降至 48.8ms（**-45%**），主要得益于输入分辨率从 640×640 降至 480×480（像素数减少约 44%）。

#### 5.4.1 640 vs 480 INT8 对比

| 指标 | 640 INT8 | 480 INT8 | 差异 |
|------|----------|----------|------|
| NPU 推理 | 89.5 ms | 48.8 ms | **-45%** |
| 预处理 | 4.7 ms | 3.3 ms | **-30%** |
| C++ 后处理 | 3.7 ms | ~2.0 ms | **-46%** |
| TCN per track | 1.3 ms | 0.5 ms | **-62%** |
| 端到端 (含 TCN×3) | 101.7 ms | **~55.6 ms** | **-45%** |

### 5.5 推理时 NPU 存储占用 — 480（≤ 20 MB ✅）

480×480 输入下中间特征图空间尺寸更小（stride 8: 60×60 vs 640 的 80×80，stride 16: 30×30 vs 40×40，stride 32: 15×15 vs 20×20），总 grid 数从 8400 降至 4725（-44%）。

#### YOLO INT8 480

| 组件 | 大小 |
|------|------|
| 权重 (INT8) | ~3.1 MB |
| 中间特征图 (INT8/FP16, 60²+30²+15² grid) | ~3-5 MB |
| 输入/输出/工作缓冲 | ~2-3 MB |
| **YOLO 小计** | **~8-11 MB** |

#### TCN FP16

| 组件 | 大小 |
|------|------|
| 权重 + 中间张量 | <1 MB |

**YOLO INT8 480 + TCN FP16 合计：~9-12 MB，≤ 20 MB ✓**

### 5.6 指标汇总（480 INT8）

| 指标 | 要求 | 480 INT8 实测 | 状态 |
|------|------|---------------|------|
| 总参数量 | ≤ 20M | ~3.15M | ✅ |
| FP32 大小 | ≤ 80 MB | ~12.0 MB | ✅ |
| 推理耗时 | ≤ 100 ms | **~55.6 ms** | ✅ |
| NPU 存储 | ≤ 20 MB | ~9-12 MB | ✅ |

> **全部四项指标达标。** 480 INT8 相比 640 INT8 推理耗时减少 45%，同时保持相同检测架构和参数量，是当前推荐的生产部署配置。

### 5.7 实测运行命令（conf=0.3, 可视化推理）

以下命令已在板端实测通过（2026-07-02）：

```bash
PYTHONPATH=src python3 scripts/infer_video_rknn_visual.py \
  --video examples/01.mp4 \
  --detector-rknn runs/final/export/yolo26n_noattn_person_int8_imgsz480_modelzoo_a043ir_450_50_rk3588.rknn \
  --temporal-rknn runs/final/export/temporal_head_yolo26n_noattn_person_640_fp16_ln_addrelu_decomposed_cliprelu_rk3588.rknn \
  --temporal-labels runs/final/export/temporal_labels_yolo26n_noattn_person_640.json \
  --output-video examples/fall_annotated_rknn_480_best.mp4 \
  --output-events examples/events_rknn_480_best.jsonl \
  --conf 0.3 \
  --imgsz 480 \
  --person-class-id 0 \
  --num-classes 1 \
  --tracker-iou-threshold 0.05 \
  --tracker-min-hits 2 \
  --tracker-max-age 30 \
  --min-fall-transition-frames 1 \
  --allow-fall-transition-alarm \
  --fall-transition-alarm-frames 1 \
  --fall-transition-score-threshold 0.0 \
  --min-fallen-seconds 0.02 \
  --allow-fallen-without-transition \
  --fallen-without-transition-seconds 0.02 \
  --fallen-score-threshold 0.0 \
  --descent-threshold 0.0 \
  --aspect-change-threshold 0.0 \
  --alarm-persist-frames 120
```

实测输出：`examples/01.mp4`（1280×720, 29.6fps, 203 frames）→ `examples/fall_annotated_rknn_480_best.mp4`, 40 alarm frames, events → `examples/events_rknn_480_best.jsonl`。
