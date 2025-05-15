# distutils: language = c++
# distutils: sources = ../src/ai_compression/c_api.cpp

from libc.stdlib cimport free
from libc.string cimport strdup
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.map cimport map
from libcpp.pair cimport pair

cdef extern from "c_api.hpp":
    ctypedef struct CortexCompressor:
        pass
    ctypedef struct CortexDecompressor:
        pass
    ctypedef struct CortexError:
        const char* message
        int code
    ctypedef struct CortexCompressionOptions:
        size_t num_threads
        int verbose
        int show_stats
        int use_delta_encoding
        int use_rle
        int compression_level

    CortexError cortex_compression_options_init(CortexCompressionOptions* options)
    CortexError cortex_compressor_create(const char* model_path, const char* format,
                                       const CortexCompressionOptions* options,
                                       CortexCompressor** handle)
    CortexError cortex_compressor_compress(CortexCompressor* handle, const char* output_path)
    CortexError cortex_compressor_get_stats(CortexCompressor* handle,
                                          size_t* original_size,
                                          size_t* compressed_size,
                                          double* compression_ratio,
                                          double* compression_time_ms)
    CortexError cortex_compressor_free(CortexCompressor* handle)
    CortexError cortex_decompressor_create(const char* compressed_path,
                                         CortexDecompressor** handle)
    CortexError cortex_decompressor_decompress(CortexDecompressor* handle,
                                             const char* output_path)
    CortexError cortex_decompressor_free(CortexDecompressor* handle)
    void cortex_error_free(CortexError* error)

class CortexError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(f"{message} (code: {code})")

def check_error(CortexError error):
    if error.code != 0:
        message = error.message.decode('utf-8') if error.message else "Unknown error"
        cortex_error_free(&error)
        raise CortexError(message, error.code)

cdef class CompressionOptions:
    cdef CortexCompressionOptions options

    def __cinit__(self):
        error = cortex_compression_options_init(&self.options)
        check_error(error)

    @property
    def num_threads(self):
        return self.options.num_threads

    @num_threads.setter
    def num_threads(self, value):
        self.options.num_threads = value

    @property
    def verbose(self):
        return bool(self.options.verbose)

    @verbose.setter
    def verbose(self, value):
        self.options.verbose = int(value)

    @property
    def show_stats(self):
        return bool(self.options.show_stats)

    @show_stats.setter
    def show_stats(self, value):
        self.options.show_stats = int(value)

    @property
    def use_delta_encoding(self):
        return bool(self.options.use_delta_encoding)

    @use_delta_encoding.setter
    def use_delta_encoding(self, value):
        self.options.use_delta_encoding = int(value)

    @property
    def use_rle(self):
        return bool(self.options.use_rle)

    @use_rle.setter
    def use_rle(self, value):
        self.options.use_rle = int(value)

    @property
    def compression_level(self):
        return self.options.compression_level

    @compression_level.setter
    def compression_level(self, value):
        self.options.compression_level = value

cdef class Compressor:
    cdef CortexCompressor* handle

    def __cinit__(self, str model_path, str format, CompressionOptions options=None):
        cdef CortexError error
        cdef const CortexCompressionOptions* options_ptr = &options.options if options else NULL
        error = cortex_compressor_create(
            model_path.encode('utf-8'),
            format.encode('utf-8'),
            options_ptr,
            &self.handle
        )
        check_error(error)

    def compress(self, str output_path):
        cdef CortexError error
        error = cortex_compressor_compress(self.handle, output_path.encode('utf-8'))
        check_error(error)

    def get_stats(self):
        cdef CortexError error
        cdef size_t original_size
        cdef size_t compressed_size
        cdef double compression_ratio
        cdef double compression_time_ms

        error = cortex_compressor_get_stats(
            self.handle,
            &original_size,
            &compressed_size,
            &compression_ratio,
            &compression_time_ms
        )
        check_error(error)

        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'compression_time_ms': compression_time_ms
        }

    def __dealloc__(self):
        if self.handle:
            error = cortex_compressor_free(self.handle)
            if error.code != 0:
                cortex_error_free(&error)

cdef class Decompressor:
    cdef CortexDecompressor* handle

    def __cinit__(self, str compressed_path):
        cdef CortexError error
        error = cortex_decompressor_create(compressed_path.encode('utf-8'), &self.handle)
        check_error(error)

    def decompress(self, str output_path):
        cdef CortexError error
        error = cortex_decompressor_decompress(self.handle, output_path.encode('utf-8'))
        check_error(error)

    def __dealloc__(self):
        if self.handle:
            error = cortex_decompressor_free(self.handle)
            if error.code != 0:
                cortex_error_free(&error)

# High-level Python interface
def compress_model(model_path: str, output_path: str, format: str = "auto",
                  num_threads: int = 1, verbose: bool = False, show_stats: bool = True,
                  use_delta_encoding: bool = True, use_rle: bool = True,
                  compression_level: int = 6) -> dict:
    """
    Compress an AI model with the specified options.
    
    Args:
        model_path: Path to the input model file
        output_path: Path to save the compressed model
        format: Model format (auto, gguf, pytorch, onnx)
        num_threads: Number of compression threads
        verbose: Enable verbose output
        show_stats: Show compression statistics
        use_delta_encoding: Enable delta encoding for sorted data
        use_rle: Enable run-length encoding
        compression_level: Compression level (1-9)
    
    Returns:
        Dictionary containing compression statistics
    """
    options = CompressionOptions()
    options.num_threads = num_threads
    options.verbose = verbose
    options.show_stats = show_stats
    options.use_delta_encoding = use_delta_encoding
    options.use_rle = use_rle
    options.compression_level = compression_level

    compressor = Compressor(model_path, format, options)
    compressor.compress(output_path)
    return compressor.get_stats()

def decompress_model(compressed_path: str, output_path: str) -> None:
    """
    Decompress a compressed AI model.
    
    Args:
        compressed_path: Path to the compressed model file
        output_path: Path to save the decompressed model
    """
    decompressor = Decompressor(compressed_path)
    decompressor.decompress(output_path) 