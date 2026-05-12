"""Generate a tiny ONNX model with realistic-shape weights for HSDR testing.

Produces a 2-layer MLP + a Conv2D layer to exercise both 2-D (Linear) and
4-D (Conv) weight tensors. Weights are FP32 sampled from N(0, 0.02), which
matches the magnitude regime of trained LLM/vision weights.

Usage: python make_test_onnx.py <out_path>
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def make_tensor(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name=name)


def main(out_path):
    rng = np.random.default_rng(0xC0FFEE)
    # Linear-style weights (2-D)
    W1 = rng.normal(0.0, 0.02, size=(512, 256)).astype(np.float32)
    b1 = np.zeros(512, dtype=np.float32)
    W2 = rng.normal(0.0, 0.02, size=(256, 512)).astype(np.float32)
    b2 = np.zeros(256, dtype=np.float32)
    # Conv2D weights (4-D): out_ch x in_ch x kH x kW.
    Wc = rng.normal(0.0, 0.05, size=(64, 32, 3, 3)).astype(np.float32)
    bc = np.zeros(64, dtype=np.float32)

    initializers = [
        make_tensor("W1", W1),
        make_tensor("b1", b1),
        make_tensor("W2", W2),
        make_tensor("b2", b2),
        make_tensor("Wc", Wc),
        make_tensor("bc", bc),
    ]

    # Inputs / outputs are just placeholders — we never run inference, we just
    # want the .onnx parser to find the initializer weights.
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 256])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 256])

    # Minimal compute graph: MatMul -> Add -> MatMul -> Add. (Conv would need
    # 4-D input; we just register Wc/bc as initializers without using them
    # in the graph. The HSDR parser still picks them up as weight segments.)
    nodes = [
        helper.make_node("MatMul", ["input", "W1"], ["h1"]),
        helper.make_node("Add", ["h1", "b1"], ["h1b"]),
        helper.make_node("MatMul", ["h1b", "W2"], ["h2"]),
        helper.make_node("Add", ["h2", "b2"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "test_mlp_with_conv_weights",
                              [inp], [out], initializer=initializers)
    model = helper.make_model(graph, producer_name="cortexsdr-test")
    # Opset that doesn't depend on newer features.
    model.opset_import[0].version = 13
    onnx.save(model, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_onnx.onnx")
