import os
import numpy as np
import pytest

from cortexsdr_sdk import (
    __version__ as sdk_version,
    CompressionOptions,
    Compressor,
    Decompressor,
    InferenceEngine,
)


def test_imports_and_version():
    assert isinstance(sdk_version, str)
    assert len(sdk_version) >= 3
    # Ensure class objects are importable
    assert CompressionOptions is not None
    assert Compressor is not None
    assert Decompressor is not None
    assert InferenceEngine is not None


def test_inference_engine_invalid_path_raises():
    with pytest.raises(Exception):
        InferenceEngine("/nonexistent/path/model.sdr")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CORTEXSDR_TEST_ONNX") is None,
    reason="Set CORTEXSDR_TEST_ONNX to run compression/integration tests",
)
def test_compress_and_infer_roundtrip(tmp_path):
    # Paths/env
    onnx_path = os.environ["CORTEXSDR_TEST_ONNX"]
    sdr_path = tmp_path / "model.sdr"

    # Compression options
    opts = CompressionOptions()
    opts.num_threads = max(1, os.cpu_count() or 1)
    opts.sparsity = 0.02
    opts.use_quantization = 1
    opts.quantization_bits = 8

    # Compress
    cmp = Compressor(onnx_path, "onnx", opts)
    cmp.compress(str(sdr_path))
    stats = cmp.stats()
    assert stats["compressed_size"] > 0
    assert os.path.exists(sdr_path)

    # Inference (dummy input vector length heuristic)
    engine = InferenceEngine(str(sdr_path))
    x = np.random.rand(128).astype(np.float32)
    y = engine.run(x)
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.size > 0


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CORTEXSDR_TEST_SDR") is None,
    reason="Set CORTEXSDR_TEST_SDR to run decompression test",
)
def test_decompress(tmp_path):
    sdr_path = os.environ["CORTEXSDR_TEST_SDR"]
    out_onnx = tmp_path / "out.onnx"

    dec = Decompressor(sdr_path, sparsity=0.02)
    dec.decompress(str(out_onnx))

    assert out_onnx.exists()
    assert out_onnx.stat().st_size > 0
