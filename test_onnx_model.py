import onnx

# Load the original model
try:
    original_model = onnx.load('onnx/examples/resources/two_transposes.onnx')
    print('Original model loaded successfully!')
    print('Model IR version:', original_model.ir_version)
    print('Producer:', original_model.producer_name)
    print('Graph name:', original_model.graph.name)
    print('Nodes:', len(original_model.graph.node))
    print('Inputs:', len(original_model.graph.input))
    print('Outputs:', len(original_model.graph.output))
except Exception as e:
    print('Error loading original model:', e)

# Load the decompressed model
try:
    decompressed_model = onnx.load('decompressed_two_transposes.onnx')
    print('\nDecompressed model loaded successfully!')
    print('Model IR version:', decompressed_model.ir_version)
    print('Producer:', decompressed_model.producer_name)
    print('Graph name:', decompressed_model.graph.name)
    print('Nodes:', len(decompressed_model.graph.node))
    print('Inputs:', len(decompressed_model.graph.input))
    print('Outputs:', len(decompressed_model.graph.output))
except Exception as e:
    print('\nError loading decompressed model:', e)
