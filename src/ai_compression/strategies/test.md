/home/mbishu/Desktop/cortexSDR/build/cortexsdr_ai_compression_cli -c /home/mbishu/Desktop/cortexSDR/mobilenetv2-7.onnx onnx /home/mbishu/Desktop/cortexSDR/mobilenetv2-7-ai-compressed.sdr
Creating compressor for model: /home/mbishu/Desktop/cortexSDR/mobilenetv2-7.onnx (format: onnx)
Using sparsity: 0.02 (2%)
Compressing model to: /home/mbishu/Desktop/cortexSDR/mobilenetv2-7-ai-compressed.sdr
ONNX Model Info: 1 inputs, 1 outputs
Processing model metadata...
Processing input: data
Processing output: mobilenetv20_output_flatten0_reshape0
Warning: Compression failed for segment 'data'. Storing uncompressed.
Compressed segment 'data': 602152 -> 602152 bytes (100%)
Warning: Compression failed for segment 'mobilenetv20_output_flatten0_reshape0'. Storing uncompressed.
Compressed segment 'mobilenetv20_output_flatten0_reshape0': 4024 -> 4024 bytes (100%)
Compression Stats:
  Original size: 606176 bytes
  Compressed size: 606176 bytes
  Compression ratio: 1:1
  Compression time: 60 ms
Compression complete.
(usr) (base) mbishu@fedora-mbishu:~/Desktop/cortexSDR$ 