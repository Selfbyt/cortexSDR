(usr) (base) mbishu@fedora-mbishu:~/Desktop/cortexSDR$ ./build.sh && build/cortexsdr_ai_compression_cli -c simple_model.onnx onnx compressed_simple_model.sdr && build/cortexsdr_ai_compression_cli -d compressed_simple_model.sdr decompressed_simple_model.onnx
Configuring project with ONNX, PyTorch, and TensorFlow support...
-- Found ONNX Protobuf library: 
-- ONNX Protobuf includes: /usr/include
-- Using local ONNX Runtime installation at /home/mbishu/Desktop/cortexSDR/onnxruntime/onnxruntime-linux-x64-gpu-1.21.1
-- ONNX Runtime library: /home/mbishu/Desktop/cortexSDR/onnxruntime/onnxruntime-linux-x64-gpu-1.21.1/lib/libonnxruntime.so
-- Using local TensorFlow installation at /home/mbishu/Desktop/cortexSDR/libtensorflow
-- Using local PyTorch installation at /home/mbishu/Desktop/cortexSDR/libtorch/libtorch
-- Configuring done (0.3s)
-- Generating done (0.1s)
-- Build files have been written to: /home/mbishu/Desktop/cortexSDR/build
[  0%] Built target cortexsdr_autogen_timestamp_deps
[  1%] Automatic MOC and UIC for target cortexsdr
[  1%] Built target cortexsdr_autogen
[  3%] Building CXX object CMakeFiles/cortexsdr.dir/src/ai_compression/api/c_api.cpp.o
[  5%] Linking CXX static library libcortexsdr.a
[ 54%] Built target cortexsdr
[ 54%] Built target cortexsdr_cli_autogen_timestamp_deps
[ 54%] Built target cortexsdr_model_converter_autogen_timestamp_deps
[ 54%] Built target test_c_api_autogen_timestamp_deps
[ 54%] Built target cortexsdr_ai_compression_cli_autogen_timestamp_deps
[ 54%] Built target sdr_test_autogen_timestamp_deps
[ 56%] Built target sdr_test_autogen
[ 60%] Built target cortexsdr_model_converter_autogen
[ 60%] Built target cortexsdr_cli_autogen
[ 61%] Built target cortexsdr_ai_compression_cli_autogen
[ 63%] Built target test_c_api_autogen
[ 65%] Linking CXX executable sdr_test
[ 70%] Linking CXX executable test_c_api
[ 70%] Linking CXX executable cortexsdr_cli
[ 70%] Linking CXX executable cortexsdr_ai_compression_cli
[ 72%] Linking CXX executable cortexsdr_model_converter
[ 78%] Built target cortexsdr_model_converter
[ 89%] Built target test_c_api
[ 89%] Built target cortexsdr_ai_compression_cli
[ 94%] Built target cortexsdr_cli
[100%] Built target sdr_test
Build complete!
Executables are in the build directory:
  - cortexsdr_cli: Main CLI tool
  - cortexsdr_model_converter: Tool to convert models to ONNX format

Remember: You can use the --sparsity or -s parameter with cortexsdr_cli to control the fraction of active bits in the SDR encoding (default 2%).
This helps tune compression and achieve higher compression ratios.
Creating compressor for model: simple_model.onnx (format: onnx)
Using sparsity: 0.02 (2%)
Compressing model to: compressed_simple_model.sdr
ONNX Model Proto Loaded. Graph: main_graph
  Initializers: 2
  Nodes: 1
  Inputs: 1
  Outputs: 1
  Added metadata segment (99 bytes).
Processing graph structure...
COMPRESSION INFO: Serialized graph structure for 'simple_model.onnx' (size 176 bytes) consists of ALL ZERO BYTES despite original graph_proto having content. Skipping segment.
Processing initializers (weights)...
  Processing initializer: linear.weight
    Added weight segment (4 bytes).
  Processing initializer: linear.bias
    Added weight segment (4 bytes).
Successfully compressed segment 'model_metadata' with strategy ID 3 (Priority: 1)

Compressing segment 'linear.weight' of type 1 with size 4 bytes
  Tensor dimensions: [1, 1]
  Sparsity ratio from metadata: 0
  Processing as tensor data
Using tensor metadata: 1 expected elements
Processing segment 'linear.weight' with 1 elements (after 0 byte header), targeting 1 active bits (2% sparsity)
Extracted 1 indices out of 1 elements (effective sparsity: 100%)
  Successfully extracted 1 indices
  Compressed 1 indices to 3 bytes
  Small segment, allowing up to 20% size increase
Successfully compressed segment 'linear.weight' with strategy ID 1 (Priority: 1)

Compressing segment 'linear.bias' of type 1 with size 4 bytes
  Tensor dimensions: [1]
  Sparsity ratio from metadata: 0
  Processing as tensor data
Using tensor metadata: 1 expected elements
Processing segment 'linear.bias' with 1 elements (after 0 byte header), targeting 1 active bits (2% sparsity)
Extracted 1 indices out of 1 elements (effective sparsity: 100%)
  Successfully extracted 1 indices
  Compressed 1 indices to 3 bytes
  Small segment, allowing up to 20% size increase
Successfully compressed segment 'linear.bias' with strategy ID 1 (Priority: 1)
Archive finalized successfully. Total Segments: 3
Compression Stats:
  Original size: 107 bytes
  Compressed size: 105 bytes
  Compression ratio: 1.01905:1
  Compression time: 1 ms
Compression complete.
Decompressing model from: compressed_simple_model.sdr to: decompressed_simple_model.onnx
Using sparsity: 0.02 (2%)
DEBUG: Expecting 3 segment headers.
DEBUG: Reading header for segment 0...
  DEBUG readString: Attempting to read length (uint16_t)...
  DEBUG readString: Read length = 14. Attempting to read 14 bytes for string...
  DEBUG readString: Successfully read string.
  DEBUG Name: model_metadata (Length: 14)
  DEBUG Type: 6
  DEBUG Strategy ID: 3
  DEBUG Original Size: 99
  DEBUG Compressed Size: 99
  DEBUG Offset: 203
  DEBUG readString: Attempting to read length (uint16_t)...
  DEBUG readString: Read length = 0. Attempting to read 0 bytes for string...
  DEBUG readString: Successfully read string.
  DEBUG Layer Name: 
  DEBUG Layer Index: 0
  DEBUG Has Metadata: false
DEBUG: Reading header for segment 1...
  DEBUG readString: Attempting to read length (uint16_t)...
  DEBUG readString: Read length = 13. Attempting to read 13 bytes for string...
  DEBUG readString: Successfully read string.
  DEBUG Name: linear.weight (Length: 13)
  DEBUG Type: 1
  DEBUG Strategy ID: 1
  DEBUG Original Size: 4
  DEBUG Compressed Size: 3
  DEBUG Offset: 302
  DEBUG readString: Attempting to read length (uint16_t)...
  DEBUG readString: Read length = 6. Attempting to read 6 bytes for string...
  DEBUG readString: Successfully read string.
  DEBUG Layer Name: weight
  DEBUG Layer Index: 0
  DEBUG Has Metadata: true
    DEBUG Num Dims: 2
      DEBUG Dim 0: 1
      DEBUG Dim 1: 1
    DEBUG Sparsity Ratio: 0
    DEBUG Is Sorted: false
DEBUG: Reading header for segment 2...
  DEBUG readString: Attempting to read length (uint16_t)...
  DEBUG readString: Read length = 11. Attempting to read 11 bytes for string...
  DEBUG readString: Successfully read string.
  DEBUG Name: linear.bias (Length: 11)
  DEBUG Type: 1
  DEBUG Strategy ID: 1
  DEBUG Original Size: 4
  DEBUG Compressed Size: 3
  DEBUG Offset: 305
  DEBUG readString: Attempting to read length (uint16_t)...
  DEBUG readString: Read length = 4. Attempting to read 4 bytes for string...
  DEBUG readString: Successfully read string.
  DEBUG Layer Name: bias
  DEBUG Layer Index: 0
  DEBUG Has Metadata: true
    DEBUG Num Dims: 1
      DEBUG Dim 0: 1
    DEBUG Sparsity Ratio: 0
    DEBUG Is Sorted: false
  Using strategy ID 3 for segment 'model_metadata' of type 6
  First 10 bytes of compressed data: 1f 8b 8 0 0 0 0 0 0 3 
  inflate returned: 1 (Z_STREAM_END)
  Decompressed 99 bytes this iteration
  First 10 bytes of decompressed data: 50 72 6f 64 75 63 65 72 3a 20 
  Using strategy ID 1 for segment 'linear.weight' of type 1
  First 10 bytes of compressed data: 1 1 0 
  First 10 bytes of decompressed data: 0 0 80 3f 
  Using strategy ID 1 for segment 'linear.bias' of type 1
  First 10 bytes of compressed data: 1 1 0 
  First 10 bytes of decompressed data: 0 0 80 3f 
Attempting ONNX model reconstruction from 3 segments...
  Warning: GRAPH_STRUCTURE_PROTO segment not found in archive. Creating default graph.
  Graph state after processing segment (or default): Nodes=0, Inputs=0, Outputs=0
  Adding 2 weight initializers and graph inputs...
  Serializing reconstructed model to: decompressed_simple_model.onnx
[libprotobuf FATAL external/com_google_protobuf/src/google/protobuf/io/zero_copy_stream_impl_lite.cc:334] CHECK failed: (count) >= (0): 
Error decompressing model: CHECK failed: (count) >= (0):  (code: 1)
(usr) (base) mbishu@fedora-mbishu:~/Desktop/cortexSDR$  elements (after 0 byte header), targeting 1 active bits (2% sparsity)
Extracted 1 indices out of 1 elements (effective sparsity: 100%)
  Successfully extracted 1 indices
  Compressed 1 indices to 3 bytes
  Small segment, allowing up to 20% size increase
Successfully compressed segment 'linear.bias' with strategy ID 1 (Priority: 1)
Archive finalized successfully. Total Segments: 3
Compression Stats:
  Original size: 107 bytes
  Compressed size: 105 bytes
  Compression ratio: 1.01905:1
  Compression time: 1 ms
Compression complete.
Decompressing model from: compressed_simple_model.sdr to: decompressed_simple_model.onnx
Using sparsity: 0.02 (2%)
(usr) (base) mbishu@fedora-mbishu:~/Desktop/cortexSDR$ :  (code: 1)/io/zero_copy_stream_impl_lite.cc:334] CHECK faile