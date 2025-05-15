#!/usr/bin/env python3
"""
Example usage of the cortexSDR Python wrapper
This demonstrates the basic functionality of the SDR class
"""

import cortexsdr
import numpy as np
import matplotlib.pyplot as plt

def main():
    print(f"CortexSDR Python Wrapper Example (v{cortexsdr.__version__})")
    print("=" * 50)
    
    # Initialize SDR with vocabulary
    vocabulary = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    sdr = cortexsdr.SDR(vocabulary)
    
    # Example 1: Text encoding
    print("\nExample 1: Text Encoding")
    text = "the quick brown fox"
    encoded = sdr.encode_text(text)
    
    print(f"Original text: {text}")
    print(f"Encoded size: {len(encoded)} bits")
    print(f"Active bits: {sdr.active_bit_count}")
    
    # Decode back to text
    decoded = sdr.decode()
    print(f"Decoded text: {decoded}")
    
    # Example 2: Number encoding
    print("\nExample 2: Number Encoding")
    number = 42.5
    encoded = sdr.encode_number(number)
    
    print(f"Original number: {number}")
    print(f"Encoded size: {len(encoded)} bits")
    print(f"Active bits: {sdr.active_bit_count}")
    
    # Decode the number
    decoded = sdr.decode()
    print(f"Decoded value: {decoded}")
    
    # Example 3: Similarity comparison
    print("\nExample 3: Similarity Comparison")
    
    # Create two similar texts
    text1 = "the quick brown fox jumps"
    text2 = "the quick brown fox leaps"
    
    # Encode both texts
    sdr1 = cortexsdr.SDR(vocabulary)
    sdr2 = cortexsdr.SDR(vocabulary)
    
    sdr1.encode_text(text1)
    sdr2.encode_text(text2)
    
    # Calculate similarity (0.0 to 1.0)
    similarity = sdr1.similarity(sdr2)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {similarity:.2f}")
    
    # Example 4: Visualization (if matplotlib is available)
    try:
        print("\nExample 4: SDR Visualization")
        
        # Get active bits
        active_bits = sdr1.get_active_bits()
        
        # Create a binary array representation
        binary_repr = np.zeros(sdr1.size)
        binary_repr[active_bits] = 1
        
        # Reshape for visualization (e.g., 50x40 grid for a 2000-bit SDR)
        grid_size = (50, 40)  # Adjust based on your SDR size
        grid = binary_repr[:grid_size[0]*grid_size[1]].reshape(grid_size)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.imshow(grid, cmap='binary', interpolation='nearest')
        plt.title(f"SDR Representation of '{text1}'")
        plt.xlabel("Bit Position (column)")
        plt.ylabel("Bit Position (row)")
        plt.colorbar(label="Bit Value")
        plt.savefig("sdr_visualization.png")
        print("Visualization saved as 'sdr_visualization.png'")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
