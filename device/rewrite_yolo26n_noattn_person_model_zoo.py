#!/usr/bin/env python3
"""Rewrite person-only YOLO26n no-attn ONNX into RKNN model-zoo outputs.

The normal Ultralytics export keeps decode/sort style tail logic in the graph.
For RK3588 deployment we expose the three raw Detect branches instead:

  box_s8,  person_s8,  score_sum_s8,
  box_s16, person_s16, score_sum_s16,
  box_s32, person_s32, score_sum_s32

This keeps MatMul/Softmax attention ops out of the backbone by construction and
removes postprocess-heavy tail ops from the RKNN graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


DEFAULT_BOX_OUTPUTS = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
]

DEFAULT_PERSON_OUTPUTS = [
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _producer_map(nodes: list[onnx.NodeProto]) -> dict[str, onnx.NodeProto]:
    producers = {}
    for node in nodes:
        for output in node.output:
            producers[output] = node
    return producers


def _collect_needed_nodes(outputs: list[str], nodes: list[onnx.NodeProto]) -> set[str]:
    producers = _producer_map(nodes)
    needed: set[str] = set()
    stack = list(outputs)
    while stack:
        tensor = stack.pop()
        node = producers.get(tensor)
        if node is None or node.name in needed:
            continue
        needed.add(node.name)
        stack.extend(inp for inp in node.input if inp)
    return needed


def _add_output(graph: onnx.GraphProto, name: str, channels: int, grid: int) -> None:
    graph.output.append(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, channels, grid, grid])
    )


def _add_scalar_initializer(graph: onnx.GraphProto, name: str, value: float) -> None:
    graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.float32), name))


def _add_int_initializer(graph: onnx.GraphProto, name: str, values: list[int]) -> None:
    graph.initializer.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--box-outputs", default=",".join(DEFAULT_BOX_OUTPUTS))
    parser.add_argument("--person-outputs", default=",".join(DEFAULT_PERSON_OUTPUTS))
    parser.add_argument("--no-person-sigmoid", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    graph = model.graph
    original_nodes = list(graph.node)
    existing_tensors = {output for node in original_nodes for output in node.output}

    box_outputs = _parse_csv(args.box_outputs)
    person_outputs = _parse_csv(args.person_outputs)
    if len(box_outputs) != 3 or len(person_outputs) != 3:
        raise ValueError("--box-outputs and --person-outputs must each contain 3 tensors")

    for tensor in box_outputs + person_outputs:
        if tensor not in existing_tensors:
            raise ValueError(f"Tensor not found in ONNX graph: {tensor}")

    grids = [args.imgsz // 8, args.imgsz // 16, args.imgsz // 32]
    strides = [8, 16, 32]
    new_nodes: list[onnx.NodeProto] = []
    output_names: list[str] = []
    final_outputs: list[tuple[str, int, int]] = []

    for stride, grid, box_tensor, person_tensor in zip(strides, grids, box_outputs, person_outputs):
        box_name = f"yolo_box_s{stride}"
        person_name = f"yolo_person_s{stride}"
        score_sum_name = f"yolo_score_sum_s{stride}"
        person_sigmoid = f"{person_name}_sigmoid"
        reduced_name = f"{score_sum_name}_reduced"

        axes_name = f"score_sum_axes_s{stride}"
        min_name = f"score_sum_min_s{stride}"
        max_name = f"score_sum_max_s{stride}"
        _add_int_initializer(graph, axes_name, [1])
        _add_scalar_initializer(graph, min_name, 0.0)
        _add_scalar_initializer(graph, max_name, 1.0)

        new_nodes.append(helper.make_node("Identity", [box_tensor], [box_name], name=f"output_box_s{stride}"))

        person_source = person_tensor
        if not args.no_person_sigmoid:
            new_nodes.append(
                helper.make_node(
                    "Sigmoid",
                    [person_tensor],
                    [person_sigmoid],
                    name=f"output_person_sigmoid_s{stride}",
                )
            )
            person_source = person_sigmoid
        new_nodes.append(
            helper.make_node("Identity", [person_source], [person_name], name=f"output_person_s{stride}")
        )
        new_nodes.append(
            helper.make_node(
                "ReduceSum",
                [person_name, axes_name],
                [reduced_name],
                name=f"output_score_sum_reduce_s{stride}",
                keepdims=1,
            )
        )
        new_nodes.append(
            helper.make_node(
                "Clip",
                [reduced_name, min_name, max_name],
                [score_sum_name],
                name=f"output_score_sum_clip_s{stride}",
            )
        )

        output_names.extend([box_name, person_name, score_sum_name])
        final_outputs.extend([(box_name, 4, grid), (person_name, 1, grid), (score_sum_name, 1, grid)])

    all_nodes = original_nodes + new_nodes
    needed = _collect_needed_nodes(output_names, all_nodes)
    kept_nodes = [node for node in all_nodes if node.name in needed]

    graph.ClearField("node")
    graph.node.extend(kept_nodes)
    graph.ClearField("output")
    for name, channels, grid in final_outputs:
        _add_output(graph, name, channels, grid)

    onnx.save(model, str(output_path))
    try:
        inferred = shape_inference.infer_shapes(onnx.load(str(output_path)), strict_mode=False)
        onnx.save(inferred, str(output_path))
        model = inferred
    except Exception as exc:
        print(f"Shape inference skipped: {exc}")
        model = onnx.load(str(output_path))

    if args.check:
        onnx.checker.check_model(model)
        ops = {node.op_type for node in model.graph.node}
        forbidden = {"MatMul", "Softmax", "TopK", "GatherElements", "NonMaxSuppression"}
        present = sorted(ops & forbidden)
        if present:
            raise AssertionError(f"Forbidden ops still present: {present}")
        attn_transpose = [
            node.name for node in model.graph.node if node.op_type == "Transpose" and "/attn/" in node.name
        ]
        if attn_transpose:
            raise AssertionError(f"Attention Transpose nodes still present: {attn_transpose}")
        if len(model.graph.output) != 9:
            raise AssertionError(f"Expected 9 outputs, got {len(model.graph.output)}")

    print(f"Saved: {output_path}")
    print(f"Nodes: {len(original_nodes)} -> {len(model.graph.node)}")
    print("Outputs:")
    for output in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in output.type.tensor_type.shape.dim]
        print(f"  {output.name}: {shape}")


if __name__ == "__main__":
    main()
