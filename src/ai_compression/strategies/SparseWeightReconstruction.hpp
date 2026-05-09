/**
 * @file SparseWeightReconstruction.hpp
 * @brief Weight reconstruction strategies from sparse representations
 * 
 * This implements several strategies to reconstruct full weight tensors
 * from sparse indices and values (SDR - Sparse Distributed Representation):
 * 1. Linear interpolation between stored values
 * 2. Statistical reconstruction using mean/std
 * 3. Nearest neighbor filling
 * 4. Smooth interpolation with exponential decay
 */

#ifndef SPARSE_WEIGHT_RECONSTRUCTION_HPP
#define SPARSE_WEIGHT_RECONSTRUCTION_HPP

#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace CortexAICompression {

enum class ReconstructionMethod {
    ZERO_FILL,              // Current method - fill with zeros
    LINEAR_INTERPOLATION,   // Interpolate between stored values
    STATISTICAL,            // Use mean/std of stored values
    NEAREST_NEIGHBOR,       // Copy nearest stored value
    SMOOTH_DECAY           // Smooth interpolation with exponential decay
};

class SparseWeightReconstruction {
public:
    /**
     * @brief Reconstruct full weight tensor from sparse indices and values
     * @param indices Indices of stored values (sorted)
     * @param values Values at those indices
     * @param totalElements Total number of elements in the tensor
     * @param method Reconstruction method to use
     * @return Reconstructed weight tensor
     */
    static std::vector<float> reconstruct(
        const std::vector<size_t>& indices,
        const std::vector<float>& values,
        size_t totalElements,
        ReconstructionMethod method = ReconstructionMethod::SMOOTH_DECAY
    );

private:
    static std::vector<float> zeroFill(
        const std::vector<size_t>& indices,
        const std::vector<float>& values,
        size_t totalElements
    );

    static std::vector<float> linearInterpolation(
        const std::vector<size_t>& indices,
        const std::vector<float>& values,
        size_t totalElements
    );

    static std::vector<float> statistical(
        const std::vector<size_t>& indices,
        const std::vector<float>& values,
        size_t totalElements
    );

    static std::vector<float> nearestNeighbor(
        const std::vector<size_t>& indices,
        const std::vector<float>& values,
        size_t totalElements
    );

    static std::vector<float> smoothDecay(
        const std::vector<size_t>& indices,
        const std::vector<float>& values,
        size_t totalElements
    );

    // Helper functions
    static float computeMean(const std::vector<float>& values);
    static float computeStd(const std::vector<float>& values, float mean);
};

} // namespace CortexAICompression

#endif // SPARSE_WEIGHT_RECONSTRUCTION_HPP
