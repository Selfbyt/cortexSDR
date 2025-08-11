#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

// SDK C API
#include "ai_compression/api/cortex_sdk.h"

namespace py = pybind11;

static void check_error(const CortexError& err) {
    if (err.code != 0) {
        std::string msg = err.message ? err.message : "Unknown error";
        throw std::runtime_error(msg + " (code: " + std::to_string(err.code) + ")");
    }
}

// RAII wrappers
class PyCompressor {
public:
    explicit PyCompressor(const std::string& model_path,
                          const std::string& format,
                          const CortexCompressionOptions& opts) {
        handle_ = nullptr;
        CortexError err = cortex_compressor_create(model_path.c_str(), format.c_str(), &opts, &handle_);
        check_error(err);
    }
    ~PyCompressor() {
        if (handle_) { cortex_compressor_free(handle_); }
    }
    void compress(const std::string& output_path) {
        check_error(cortex_compressor_compress(handle_, output_path.c_str()));
    }
    py::dict stats() {
        size_t orig=0, comp=0; double ratio=0.0, ms=0.0;
        check_error(cortex_compressor_get_stats(handle_, &orig, &comp, &ratio, &ms));
        py::dict d;
        d["original_size"] = py::int_(orig);
        d["compressed_size"] = py::int_(comp);
        d["compression_ratio"] = ratio;
        d["compression_time_ms"] = ms;
        return d;
    }
private:
    CortexCompressorHandle handle_;
};

class PyDecompressor {
public:
    explicit PyDecompressor(const std::string& compressed_path, float sparsity) {
        handle_ = nullptr;
        check_error(cortex_decompressor_create(compressed_path.c_str(), &handle_, sparsity));
        path_ = compressed_path;
    }
    ~PyDecompressor() {
        if (handle_) { cortex_decompressor_free(handle_); }
    }
    void decompress(const std::string& output_path) {
        check_error(cortex_decompressor_decompress(handle_, path_.c_str(), output_path.c_str()));
    }
private:
    CortexDecompressorHandle handle_;
    std::string path_;
};

class PyInferenceEngine {
public:
    explicit PyInferenceEngine(const std::string& compressed_model_path) {
        handle_ = nullptr;
        if (!std::filesystem::exists(compressed_model_path)) {
            throw std::runtime_error("Compressed model path does not exist: " + compressed_model_path);
        }
        check_error(cortex_inference_engine_create(compressed_model_path.c_str(), &handle_));
        if (!handle_) {
            throw std::runtime_error("Failed to create inference engine (null handle)");
        }
    }
    ~PyInferenceEngine() {
        if (handle_) { cortex_inference_engine_free(handle_); }
    }
    void set_batch_size(size_t batch_size) {
        check_error(cortex_inference_engine_set_batch_size(handle_, batch_size));
    }
    void enable_dropout(bool enable) {
        check_error(cortex_inference_engine_enable_dropout(handle_, enable ? 1 : 0));
    }
    void set_mode(int training_mode) {
        check_error(cortex_inference_engine_set_mode(handle_, training_mode));
    }
    py::array_t<float> run_layer(const std::string& layer_name,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> input) {
        py::buffer_info in_info = input.request();
        size_t in_size = static_cast<size_t>(in_info.size);
        const float* in_ptr = static_cast<const float*>(in_info.ptr);

        size_t out_cap = std::max<size_t>(4096, in_size * 4);
        std::vector<float> out(out_cap);
        size_t actual_size = 0;
        CortexError err = cortex_inference_engine_run_layer(
            handle_, layer_name.c_str(), in_ptr, in_size, out.data(), out_cap, &actual_size
        );
        check_error(err);
        return py::array_t<float>(static_cast<py::ssize_t>(actual_size), out.data()).attr("copy")();
    }
    py::array_t<float> run(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
        py::buffer_info in_info = input.request();
        size_t in_size = static_cast<size_t>(in_info.size);
        const float* in_ptr = static_cast<const float*>(in_info.ptr);

        // Allocate a generous output buffer (at least 4096 or 4x input)
        size_t out_cap = std::max<size_t>(4096, in_size * 4);
        std::vector<float> out(out_cap);
        size_t actual_size = 0;
        CortexError err = cortex_inference_engine_run(
            handle_,
            in_ptr,
            in_size,
            out.data(),
            out_cap,
            &actual_size
        );
        check_error(err);
        return py::array_t<float>(static_cast<py::ssize_t>(actual_size), out.data()).attr("copy")();
    }
private:
    CortexInferenceEngineHandle handle_;
};

PYBIND11_MODULE(_bindings, m) {
    m.doc() = "Python bindings for CortexSDR SDK (compression, decompression, inference)";

    py::class_<CortexCompressionOptions>(m, "CompressionOptions")
        .def(py::init<>())
        .def_readwrite("num_threads", &CortexCompressionOptions::num_threads)
        .def_readwrite("verbose", &CortexCompressionOptions::verbose)
        .def_readwrite("show_stats", &CortexCompressionOptions::show_stats)
        .def_readwrite("use_delta_encoding", &CortexCompressionOptions::use_delta_encoding)
        .def_readwrite("use_rle", &CortexCompressionOptions::use_rle)
        .def_readwrite("compression_level", &CortexCompressionOptions::compression_level)
        .def_readwrite("use_quantization", &CortexCompressionOptions::use_quantization)
        .def_readwrite("quantization_bits", &CortexCompressionOptions::quantization_bits)
        .def_readwrite("sparsity", &CortexCompressionOptions::sparsity);

    py::class_<PyCompressor>(m, "Compressor")
        .def(py::init<const std::string&, const std::string&, const CortexCompressionOptions&>(),
             py::arg("model_path"), py::arg("format"), py::arg("options"))
        .def("compress", &PyCompressor::compress)
        .def("stats", &PyCompressor::stats);

    py::class_<PyDecompressor>(m, "Decompressor")
        .def(py::init<const std::string&, float>(), py::arg("compressed_path"), py::arg("sparsity") = 0.02f)
        .def("decompress", &PyDecompressor::decompress);

    py::class_<PyInferenceEngine>(m, "InferenceEngine")
        .def(py::init<const std::string&>(), py::arg("compressed_model_path"))
        .def("set_batch_size", &PyInferenceEngine::set_batch_size, py::arg("batch_size"))
        .def("enable_dropout", &PyInferenceEngine::enable_dropout, py::arg("enable"))
        .def("set_mode", &PyInferenceEngine::set_mode, py::arg("training_mode"))
        .def("run", &PyInferenceEngine::run, py::arg("input"))
        .def("run_layer", &PyInferenceEngine::run_layer, py::arg("layer_name"), py::arg("input"));
}
