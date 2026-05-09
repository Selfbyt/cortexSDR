/**
 * @file SparseWeightReconstruction.cpp
 * @brief Implementation of weight reconstruction from sparse representations
 */

#include "SparseWeightReconstruction.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>

namespace CortexAICompression {

std::vector<float> SparseWeightReconstruction::reconstruct(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t totalElements,
    ReconstructionMethod method
) {
    switch (method) {
        case ReconstructionMethod::ZERO_FILL:
            return zeroFill(indices, values, totalElements);
        case ReconstructionMethod::LINEAR_INTERPOLATION:
            return linearInterpolation(indices, values, totalElements);
        case ReconstructionMethod::STATISTICAL:
            return statistical(indices, values, totalElements);
        case ReconstructionMethod::NEAREST_NEIGHBOR:
            return nearestNeighbor(indices, values, totalElements);
        case ReconstructionMethod::SMOOTH_DECAY:
            return smoothDecay(indices, values, totalElements);
        default:
            return zeroFill(indices, values, totalElements);
    }
}

std::vector<float> SparseWeightReconstruction::zeroFill(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t totalElements
) {
    std::vector<float> result(totalElements, 0.0f);
    
    for (size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        if (indices[i] < totalElements) {
            result[indices[i]] = values[i];
        }
    }
    
    return result;
}

std::vector<float> SparseWeightReconstruction::linearInterpolation(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t totalElements
) {
    std::vector<float> result(totalElements, 0.0f);
    
    if (indices.empty() || values.empty()) {
        return result;
    }
    
    // First, set the known values
    for (size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        if (indices[i] < totalElements) {
            result[indices[i]] = values[i];
        }
    }
    
    // Interpolate between known values
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        size_t start_idx = indices[i];
        size_t end_idx = indices[i + 1];
        
        if (start_idx >= totalElements || end_idx >= totalElements) continue;
        
        float start_val = values[i];
        float end_val = values[i + 1];
        
        // Linear interpolation for values between start and end
        for (size_t j = start_idx + 1; j < end_idx; ++j) {
            float t = static_cast<float>(j - start_idx) / static_cast<float>(end_idx - start_idx);
            result[j] = start_val + t * (end_val - start_val);
        }
    }
    
    // Fill values before first index with first value (scaled down)
    if (indices[0] > 0) {
        float first_val = values[0] * 0.5f; // Scale down for safety
        for (size_t i = 0; i < indices[0]; ++i) {
            result[i] = first_val;
        }
    }
    
    // Fill values after last index with last value (scaled down)
    if (indices.back() < totalElements - 1) {
        float last_val = values.back() * 0.5f; // Scale down for safety
        for (size_t i = indices.back() + 1; i < totalElements; ++i) {
            result[i] = last_val;
        }
    }
    
    return result;
}

std::vector<float> SparseWeightReconstruction::statistical(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t totalElements
) {
    std::vector<float> result(totalElements, 0.0f);
    
    if (indices.empty() || values.empty()) {
        return result;
    }
    
    // Calculate statistics of stored values
    float mean = computeMean(values);
    float std = computeStd(values, mean);
    
    // Set known values
    for (size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        if (indices[i] < totalElements) {
            result[indices[i]] = values[i];
        }
    }
    
    // Fill missing values with samples from normal distribution
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(mean, std);
    
    for (size_t i = 0; i < totalElements; ++i) {
        // Check if this index is already set
        bool is_set = false;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (indices[j] == i) {
                is_set = true;
                break;
            }
        }
        
        if (!is_set) {
            // Sample from distribution, but scale down to be conservative
            result[i] = dist(gen) * 0.3f; // Scale down to reduce error
        }
    }
    
    return result;
}

std::vector<float> SparseWeightReconstruction::nearestNeighbor(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t totalElements
) {
    std::vector<float> result(totalElements, 0.0f);
    
    if (indices.empty() || values.empty()) {
        return result;
    }
    
    // For each position, find the nearest stored value
    for (size_t i = 0; i < totalElements; ++i) {
        // Find nearest index
        size_t nearest_idx = 0;
        size_t min_distance = totalElements;
        
        for (size_t j = 0; j < indices.size(); ++j) {
            size_t distance = (i > indices[j]) ? (i - indices[j]) : (indices[j] - i);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_idx = j;
            }
        }
        
        // Use the nearest value, but scale it down based on distance
        float distance_factor = 1.0f / (1.0f + std::sqrt(static_cast<float>(min_distance)));
        result[i] = values[nearest_idx] * distance_factor;
    }
    
    return result;
}

std::vector<float> SparseWeightReconstruction::smoothDecay(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t totalElements
) {
    std::vector<float> result(totalElements, 0.0f);
    
    if (indices.empty() || values.empty()) {
        return result;
    }
    
    // Set known values
    for (size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        if (indices[i] < totalElements) {
            result[indices[i]] = values[i];
        }
    }
    
    // For each missing position, compute weighted average of nearby stored values
    const size_t MAX_NEIGHBORS = 5;
    const float DECAY_RATE = 0.1f; // How quickly influence decays with distance
    
    for (size_t i = 0; i < totalElements; ++i) {
        // Check if this position already has a value
        bool has_value = false;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (indices[j] == i) {
                has_value = true;
                break;
            }
        }
        
        if (has_value) continue;
        
        // Find nearest neighbors
        std::vector<std::pair<size_t, float>> neighbors; // (distance, value)
        
        for (size_t j = 0; j < indices.size(); ++j) {
            size_t distance = (i > indices[j]) ? (i - indices[j]) : (indices[j] - i);
            neighbors.push_back({distance, values[j]});
        }
        
        // Sort by distance
        std::sort(neighbors.begin(), neighbors.end());
        
        // Take top MAX_NEIGHBORS
        size_t num_neighbors = std::min(MAX_NEIGHBORS, neighbors.size());
        
        // Compute weighted average with exponential decay
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (size_t j = 0; j < num_neighbors; ++j) {
            float distance = static_cast<float>(neighbors[j].first);
            float weight = std::exp(-DECAY_RATE * distance);
            weighted_sum += weight * neighbors[j].second;
            weight_sum += weight;
        }
        
        if (weight_sum > 0.0f) {
            result[i] = weighted_sum / weight_sum;
        }
    }
    
    return result;
}

float SparseWeightReconstruction::computeMean(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / values.size();
}

float SparseWeightReconstruction::computeStd(const std::vector<float>& values, float mean) {
    if (values.empty()) return 0.0f;
    
    float sum_squared_diff = 0.0f;
    for (float val : values) {
        float diff = val - mean;
        sum_squared_diff += diff * diff;
    }
    
    return std::sqrt(sum_squared_diff / values.size());
}

} // namespace CortexAICompression
