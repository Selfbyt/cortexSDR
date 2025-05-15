import onnx
from onnx import checker # Explicitly import checker
import sys

def check_model(model_path):
    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        print(f"Successfully loaded ONNX model: '{model_path}'")

        # Check the model's validity
        checker.check_model(model) # Use the imported checker
        print(f"ONNX model check passed.")

        # Print basic graph info
        graph = model.graph
        print(f"--- Graph Info ---")
        print(f"Name: {graph.name}")
        print(f"Nodes: {len(graph.node)}")
        print(f"Inputs: {len(graph.input)}")
        print(f"Outputs: {len(graph.output)}")
        print(f"Initializers: {len(graph.initializer)}")
        print(f"------------------")

        # Optional: Print initializer names and shapes for more detail
        # print("Initializers:")
        # for init in graph.initializer:
        #     print(f"  - {init.name}: shape={list(init.dims)}")

    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)
    except checker.ValidationError as e: # Use the imported checker for the exception type
        print(f"ONNX model validation failed for '{model_path}':\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing ONNX model '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_onnx.py <path_to_onnx_model>", file=sys.stderr)
        sys.exit(1)
    
    model_path_arg = sys.argv[1]
    check_model(model_path_arg)
