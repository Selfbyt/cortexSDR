import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

def compress_model(model_path, output_path, sparsity=None):
    """
    Compress a model using cortexsdr_ai_compression_cli with optional sparsity parameter
    """
    cmd = ["build/cortexsdr_ai_compression_cli", "-c", model_path, "onnx", output_path]
    
    # Add sparsity parameter if specified
    if sparsity is not None:
        cmd.append(str(sparsity))
    
    # Start timing
    start_time = time.time()
    
    # Run compression
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # End timing
    elapsed_time = time.time() - start_time
    
    # Check if compression was successful
    if process.returncode != 0:
        print(f"Error compressing model: {stderr}")
        return None
    
    # Parse compression stats from output
    original_size = None
    compressed_size = None
    compression_ratio = None
    
    for line in stdout.split('\n'):
        if "Original size:" in line:
            original_size = int(line.split(":")[1].strip().split()[0])
        elif "Compressed size:" in line:
            compressed_size = int(line.split(":")[1].strip().split()[0])
        elif "Compression ratio:" in line:
            compression_ratio = float(line.split(":")[1].strip().split('x')[0])
    
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "elapsed_time": elapsed_time,
        "sparsity": sparsity if sparsity is not None else 0.02  # Default is 2%
    }

def decompress_model(compressed_path, output_path, sparsity=None):
    """
    Decompress a model using cortexsdr_ai_compression_cli
    """
    cmd = ["build/cortexsdr_ai_compression_cli", "-d", compressed_path, output_path]
    
    # Add sparsity parameter if specified
    if sparsity is not None:
        cmd.append(str(sparsity))
    
    # Start timing
    start_time = time.time()
    
    # Run decompression
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # End timing
    elapsed_time = time.time() - start_time
    
    # Check if decompression was successful
    if process.returncode != 0:
        print(f"Error decompressing model: {stderr}")
        return None
    
    return {
        "elapsed_time": elapsed_time,
        "sparsity": sparsity if sparsity is not None else 0.02  # Default is 2%
    }

def test_different_sparsity_values(model_path, sparsity_values):
    """
    Test compression with different sparsity values and plot results
    """
    results = []
    
    for sparsity in sparsity_values:
        print(f"\nTesting with sparsity = {sparsity} ({sparsity*100}%)")
        output_path = f"compressed_gpt2_{int(sparsity*100)}.sdr"
        
        # Compress model
        compression_result = compress_model(model_path, output_path, sparsity)
        
        if compression_result:
            # Get file sizes
            original_file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            compressed_file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            # Calculate actual compression ratio
            actual_ratio = original_file_size / compressed_file_size if compressed_file_size > 0 else 0
            
            # Add to results
            results.append({
                "sparsity": sparsity,
                "original_size_mb": original_file_size,
                "compressed_size_mb": compressed_file_size,
                "compression_ratio": actual_ratio,
                "compression_time": compression_result["elapsed_time"]
            })
            
            print(f"Original size: {original_file_size:.2f} MB")
            print(f"Compressed size: {compressed_file_size:.2f} MB")
            print(f"Compression ratio: {actual_ratio:.2f}:1")
            print(f"Compression time: {compression_result['elapsed_time']:.2f} seconds")
        
    return results

def plot_results(results):
    """
    Plot compression results
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    sparsity_values = [r["sparsity"] * 100 for r in results]  # Convert to percentage
    compression_ratios = [r["compression_ratio"] for r in results]
    compression_times = [r["compression_time"] / 60 for r in results]  # Convert to minutes
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot compression ratio vs sparsity
    ax1.plot(sparsity_values, compression_ratios, 'o-', color='blue')
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('Compression Ratio vs Sparsity')
    ax1.grid(True)
    
    # Plot compression time vs sparsity
    ax2.plot(sparsity_values, compression_times, 'o-', color='red')
    ax2.set_xlabel('Sparsity (%)')
    ax2.set_ylabel('Compression Time (minutes)')
    ax2.set_title('Compression Time vs Sparsity')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('gpt2_compression_results.png')
    print("Results saved to gpt2_compression_results.png")

if __name__ == "__main__":
    # Model path
    model_path = "gpt2-10.onnx"
    
    # Test with different sparsity values
    # The default is 0.02 (2%), let's try a range around that
    sparsity_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    
    # Run tests
    print(f"Testing GPT-2 compression with different sparsity values")
    results = test_different_sparsity_values(model_path, sparsity_values)
    
    # Plot results
    if results:
        plot_results(results)
