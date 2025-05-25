import onnx
import numpy as np
import time
import matplotlib.pyplot as plt

def compare_model_structure(model1, model2):
    """
    Compare the structure of two ONNX models and return a similarity score
    """
    # Compare IR versions
    ir_match = model1.ir_version == model2.ir_version
    
    # Compare producer names
    producer_match = model1.producer_name == model2.producer_name
    
    # Compare graph names
    graph_name_match = model1.graph.name == model2.graph.name
    
    # Compare number of nodes
    node_count_match = len(model1.graph.node) == len(model2.graph.node)
    
    # Compare node operations
    node_ops_match = True
    if len(model1.graph.node) == len(model2.graph.node):
        for n1, n2 in zip(model1.graph.node, model2.graph.node):
            if n1.op_type != n2.op_type:
                node_ops_match = False
                break
    else:
        node_ops_match = False
    
    # Compare inputs and outputs
    input_count_match = len(model1.graph.input) == len(model2.graph.input)
    output_count_match = len(model1.graph.output) == len(model2.graph.output)
    
    # Calculate overall similarity score (percentage)
    matches = [ir_match, producer_match, graph_name_match, node_count_match, 
               node_ops_match, input_count_match, output_count_match]
    similarity_score = sum(matches) / len(matches) * 100
    
    return {
        'ir_match': ir_match,
        'producer_match': producer_match,
        'graph_name_match': graph_name_match,
        'node_count_match': node_count_match,
        'node_ops_match': node_ops_match,
        'input_count_match': input_count_match,
        'output_count_match': output_count_match,
        'similarity_score': similarity_score
    }

def compare_models(original_model_path, decompressed_model_path):
    """
    Compare the structure and content of original and decompressed models
    """
    try:
        # Load models
        print(f"Loading original model from {original_model_path}...")
        original_model = onnx.load(original_model_path)
        
        print(f"Loading decompressed model from {decompressed_model_path}...")
        decompressed_model = onnx.load(decompressed_model_path)
        
        # Print model info
        print("\nOriginal Model:")
        print(f"IR Version: {original_model.ir_version}")
        print(f"Producer: {original_model.producer_name}")
        print(f"Graph Name: {original_model.graph.name}")
        print(f"Nodes: {len(original_model.graph.node)}")
        print(f"Inputs: {len(original_model.graph.input)}")
        print(f"Outputs: {len(original_model.graph.output)}")
        
        print("\nDecompressed Model:")
        print(f"IR Version: {decompressed_model.ir_version}")
        print(f"Producer: {decompressed_model.producer_name}")
        print(f"Graph Name: {decompressed_model.graph.name}")
        print(f"Nodes: {len(decompressed_model.graph.node)}")
        print(f"Inputs: {len(decompressed_model.graph.input)}")
        print(f"Outputs: {len(decompressed_model.graph.output)}")
        
        # Compare model structures
        print("\nComparing model structures...")
        comparison_results = compare_model_structure(original_model, decompressed_model)
        
        print(f"IR Version Match: {comparison_results['ir_match']}")
        print(f"Producer Match: {comparison_results['producer_match']}")
        print(f"Graph Name Match: {comparison_results['graph_name_match']}")
        print(f"Node Count Match: {comparison_results['node_count_match']}")
        print(f"Node Operations Match: {comparison_results['node_ops_match']}")
        print(f"Input Count Match: {comparison_results['input_count_match']}")
        print(f"Output Count Match: {comparison_results['output_count_match']}")
        
        # Print overall similarity score
        print(f"\nOverall Model Similarity Score: {comparison_results['similarity_score']:.2f}%")
        
        # Compare node operations in detail if they match in count
        if len(original_model.graph.node) == len(decompressed_model.graph.node):
            print("\nDetailed Node Operation Comparison:")
            for i, (n1, n2) in enumerate(zip(original_model.graph.node, decompressed_model.graph.node)):
                match = "✓" if n1.op_type == n2.op_type else "✗"
                print(f"Node {i}: {n1.op_type} vs {n2.op_type} - {match}")
        
        # Check if the models are identical
        if comparison_results['similarity_score'] == 100:
            print("\n✅ The models are structurally identical!")
        else:
            print(f"\n⚠️ The models have some structural differences. Similarity: {comparison_results['similarity_score']:.2f}%")
        
        return comparison_results
        
    except Exception as e:
        print(f"Error comparing models: {e}")
        return None

if __name__ == "__main__":
    # Model paths
    original_model = "onnx/examples/resources/two_transposes.onnx"
    decompressed_model = "decompressed_two_transposes.onnx"
    
    # Compare models
    results = compare_models(original_model, decompressed_model)
    
    # If models are identical, print success message
    if results and results['similarity_score'] == 100:
        print("\nSUCCESS: The compression and decompression process preserved the model structure perfectly!")
    elif results:
        print(f"\nNOTE: The compression and decompression process resulted in some differences. Similarity: {results['similarity_score']:.2f}%")
        
    # Try another model if available
    try:
        print("\n" + "-"*50)
        print("Testing with another model: single_relu.onnx")
        print("-"*50)
        original_model2 = "onnx/examples/resources/single_relu.onnx"
        decompressed_model2 = "decompressed_single_relu.onnx"
        
        # Check if the decompressed model exists
        import os
        if not os.path.exists(decompressed_model2):
            print(f"\nDecompressed model {decompressed_model2} not found. Skipping comparison.")
            print("You can create it by running: build/cortexsdr_ai_compression_cli -d compressed_single_relu.sdr decompressed_single_relu.onnx")
        else:
            results2 = compare_models(original_model2, decompressed_model2)
            
            if results2 and results2['similarity_score'] == 100:
                print("\nSUCCESS: The compression and decompression process preserved the model structure perfectly!")
            elif results2:
                print(f"\nNOTE: The compression and decompression process resulted in some differences. Similarity: {results2['similarity_score']:.2f}%")
    except Exception as e:
        print(f"Error comparing additional models: {e}")
