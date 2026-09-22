#!/usr/bin/env python3
"""Decompose ONNX ops that are unsupported by RKNN runtime 1.6.0.

RKNN runtime 1.6.0 does not support ONNX LayerNormalization. GeometryTCN uses
LayerNorm over the last hidden dimension, so each LayerNormalization node can be
expanded into basic ONNX ops:

  mean = ReduceMean(x)
  centered = x - mean
  var = ReduceMean(centered * centered)
  norm = centered / Sqrt(var + eps)
  y = norm * gamma + beta

Some toolkit/runtime combinations may also preserve or fuse an AddRelu op. The
RK3588 librknnrt.so 1.6.0 / driver 0.9.8 stack does not support it reliably, so
this script rewrites AddRelu(x, y) into Add(x, y) followed by Relu.

RKNN-Toolkit2 1.6.0 may fuse Add + Relu back into internal AddRelu/ConvAddRelu
during build. Use --add-relu-as-clip to rewrite Add -> Relu patterns as
Add -> Clip(min=0), which is equivalent to ReLU and prevents that fusion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def _attr(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def _make_scalar_initializer(name: str, value: float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name=name)


def decompose_layernorm(model: onnx.ModelProto) -> int:
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    replaced = 0

    for node in model.graph.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue

        if len(node.input) < 2:
            raise ValueError(f"LayerNormalization node {node.name!r} has too few inputs")

        axis = int(_attr(node, "axis", -1))
        if axis not in (-1,):
            raise ValueError(
                f"Only axis=-1 LayerNormalization is supported, got axis={axis} at {node.name!r}"
            )

        epsilon = float(_attr(node, "epsilon", 1e-5))
        x = node.input[0]
        gamma = node.input[1]
        beta = node.input[2] if len(node.input) >= 3 and node.input[2] else None
        y = node.output[0]
        raw_prefix = (node.name or f"LayerNorm_{replaced}").strip("/").replace("/", "_")
        prefix = raw_prefix.replace("LayerNormalization", "LayerNormDecomposed")

        mean_0 = f"{prefix}_mean0"
        mean = f"{prefix}_mean"
        centered = f"{prefix}_centered"
        squared = f"{prefix}_squared"
        var_0 = f"{prefix}_var0"
        var = f"{prefix}_var"
        eps_name = f"{prefix}_eps"
        var_eps = f"{prefix}_var_eps"
        denom = f"{prefix}_denom"
        norm = f"{prefix}_norm"
        scaled = f"{prefix}_scaled"

        new_initializers.append(_make_scalar_initializer(eps_name, epsilon))

        new_nodes.extend(
            [
                helper.make_node(
                    "ReduceMean",
                    [x],
                    [mean_0],
                    name=f"{prefix}_ReduceMean",
                    axes=[-1],
                    keepdims=0,
                ),
                helper.make_node(
                    "Unsqueeze",
                    [mean_0],
                    [mean],
                    name=f"{prefix}_UnsqueezeMean",
                    axes=[-1],
                ),
                helper.make_node(
                    "Sub",
                    [x, mean],
                    [centered],
                    name=f"{prefix}_SubMean",
                ),
                helper.make_node(
                    "Mul",
                    [centered, centered],
                    [squared],
                    name=f"{prefix}_Square",
                ),
                helper.make_node(
                    "ReduceMean",
                    [squared],
                    [var_0],
                    name=f"{prefix}_ReduceVar",
                    axes=[-1],
                    keepdims=0,
                ),
                helper.make_node(
                    "Unsqueeze",
                    [var_0],
                    [var],
                    name=f"{prefix}_UnsqueezeVar",
                    axes=[-1],
                ),
                helper.make_node(
                    "Add",
                    [var, eps_name],
                    [var_eps],
                    name=f"{prefix}_AddEps",
                ),
                helper.make_node(
                    "Sqrt",
                    [var_eps],
                    [denom],
                    name=f"{prefix}_Sqrt",
                ),
                helper.make_node(
                    "Div",
                    [centered, denom],
                    [norm],
                    name=f"{prefix}_Div",
                ),
                helper.make_node(
                    "Mul",
                    [norm, gamma],
                    [scaled],
                    name=f"{prefix}_Scale",
                ),
            ]
        )

        if beta is None:
            new_nodes.append(helper.make_node("Identity", [scaled], [y], name=f"{prefix}_Out"))
        else:
            new_nodes.append(helper.make_node("Add", [scaled, beta], [y], name=f"{prefix}_Bias"))

        replaced += 1

    if replaced == 0:
        return 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    return replaced


def decompose_addrelu(model: onnx.ModelProto) -> int:
    new_nodes: list[onnx.NodeProto] = []
    replaced = 0

    for node in model.graph.node:
        if node.op_type != "AddRelu":
            new_nodes.append(node)
            continue

        if len(node.input) != 2:
            raise ValueError(f"AddRelu node {node.name!r} must have exactly 2 inputs")
        if len(node.output) != 1:
            raise ValueError(f"AddRelu node {node.name!r} must have exactly 1 output")

        raw_prefix = (node.name or f"AddRelu_{replaced}").strip("/").replace("/", "_")
        prefix = raw_prefix.replace("AddRelu", "AddReluDecomposed")
        add_out = f"{prefix}_add"

        new_nodes.extend(
            [
                helper.make_node(
                    "Add",
                    list(node.input),
                    [add_out],
                    name=f"{prefix}_Add",
                ),
                helper.make_node(
                    "Relu",
                    [add_out],
                    [node.output[0]],
                    name=f"{prefix}_Relu",
                ),
            ]
        )
        replaced += 1

    if replaced == 0:
        return 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return replaced


def rewrite_add_relu_as_clip(model: onnx.ModelProto) -> int:
    producers = {output: node for node in model.graph.node for output in node.output}
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    existing_initializers = {initializer.name for initializer in model.graph.initializer}
    min_name = "edgefall_clip_relu_min_zero"
    max_name = "edgefall_clip_relu_max"
    replaced = 0

    if min_name not in existing_initializers:
        new_initializers.append(numpy_helper.from_array(np.array(0, dtype=np.float32), min_name))
    if max_name not in existing_initializers:
        new_initializers.append(
            numpy_helper.from_array(np.array(np.finfo(np.float32).max, dtype=np.float32), max_name)
        )

    for node in model.graph.node:
        if (
            node.op_type == "Relu"
            and node.input
            and node.input[0] in producers
            and producers[node.input[0]].op_type == "Add"
        ):
            name = f"{node.name or f'Relu_{replaced}'}_ClipNoFuse"
            new_nodes.append(
                helper.make_node("Clip", [node.input[0], min_name, max_name], list(node.output), name=name)
            )
            replaced += 1
            continue

        new_nodes.append(node)

    if replaced == 0:
        return 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    return replaced


def compare_outputs(input_path: Path, output_path: Path) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 16, 10), dtype=np.float32)

    sess_a = ort.InferenceSession(str(input_path), providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    out_a = sess_a.run(None, {"geometry": x})[0]
    out_b = sess_b.run(None, {"geometry": x})[0]
    return float(np.max(np.abs(out_a - out_b)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace RKNN 1.6.0-unsupported ONNX ops with basic ops"
    )
    parser.add_argument("--input", required=True, help="Input ONNX path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run ONNX checker and compare original/decomposed outputs with ONNX Runtime",
    )
    parser.add_argument(
        "--add-relu-as-clip",
        action="store_true",
        help="Rewrite Add->Relu patterns as Add->Clip(min=0) to prevent RKNN AddRelu fusion",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    replaced_ln = decompose_layernorm(model)
    replaced_addrelu = decompose_addrelu(model)
    replaced_add_relu_clip = rewrite_add_relu_as_clip(model) if args.add_relu_as_clip else 0
    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    onnx.save(model, str(output_path))

    ops = sorted({node.op_type for node in model.graph.node})
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Replaced LayerNormalization nodes: {replaced_ln}")
    print(f"Replaced AddRelu nodes: {replaced_addrelu}")
    print(f"Replaced Add->Relu patterns with Clip: {replaced_add_relu_clip}")
    print(f"Ops: {ops}")
    if "LayerNormalization" in ops:
        raise RuntimeError("LayerNormalization still exists after decomposition")
    if "AddRelu" in ops:
        raise RuntimeError("AddRelu still exists after decomposition")

    if args.check:
        max_abs_diff = compare_outputs(input_path, output_path)
        print(f"ONNX Runtime max_abs_diff: {max_abs_diff:.8g}")
        if max_abs_diff > 1e-4:
            raise RuntimeError(f"Output diff too large: {max_abs_diff}")


if __name__ == "__main__":
    main()
