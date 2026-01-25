/**
 * @file kv_cache.hpp
 * @brief Key-Value cache for efficient transformer inference
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <cstddef>
#include <memory>
#include <mutex>

namespace CortexAICompression {
namespace Utils {

/**
 * @brief KV cache entry for a single layer
 */
struct KVCacheEntry {
    std::vector<float> keys;      // Cached key states
    std::vector<float> values;    // Cached value states
    size_t seq_len;               // Current sequence length
    size_t max_seq_len;           // Maximum capacity
    size_t hidden_dim;            // Hidden dimension
    
    KVCacheEntry(size_t max_len, size_t hidden)
        : seq_len(0), max_seq_len(max_len), hidden_dim(hidden) {
        keys.reserve(max_len * hidden);
        values.reserve(max_len * hidden);
    }
    
    void reset() {
        keys.clear();
        values.clear();
        seq_len = 0;
    }
    
    bool append(const float* new_keys, const float* new_values, size_t new_len) {
        if (seq_len + new_len > max_seq_len) {
            return false;  // Cache overflow
        }
        
        size_t append_size = new_len * hidden_dim;
        keys.insert(keys.end(), new_keys, new_keys + append_size);
        values.insert(values.end(), new_values, new_values + append_size);
        seq_len += new_len;
        return true;
    }
};

/**
 * @brief Multi-layer KV cache manager for transformer models
 */
class KVCacheManager {
public:
    KVCacheManager(size_t num_layers, size_t max_seq_len, size_t hidden_dim)
        : num_layers_(num_layers), max_seq_len_(max_seq_len), hidden_dim_(hidden_dim) {
        for (size_t i = 0; i < num_layers; ++i) {
            caches_.emplace_back(std::make_unique<KVCacheEntry>(max_seq_len, hidden_dim));
        }
    }
    
    KVCacheEntry* get_cache(size_t layer_idx) {
        if (layer_idx >= caches_.size()) return nullptr;
        return caches_[layer_idx].get();
    }
    
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& cache : caches_) {
            cache->reset();
        }
    }
    
    void reset_layer(size_t layer_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (layer_idx < caches_.size()) {
            caches_[layer_idx]->reset();
        }
    }
    
    size_t get_current_seq_len(size_t layer_idx) const {
        if (layer_idx >= caches_.size()) return 0;
        return caches_[layer_idx]->seq_len;
    }
    
    bool is_cache_full(size_t layer_idx) const {
        if (layer_idx >= caches_.size()) return false;
        return caches_[layer_idx]->seq_len >= caches_[layer_idx]->max_seq_len;
    }
    
private:
    size_t num_layers_;
    size_t max_seq_len_;
    size_t hidden_dim_;
    std::vector<std::unique_ptr<KVCacheEntry>> caches_;
    std::mutex mutex_;
};

/**
 * @brief Paged attention cache for efficient memory management
 */
class PagedKVCache {
public:
    static constexpr size_t PAGE_SIZE = 256;  // tokens per page
    
    struct Page {
        std::vector<float> keys;
        std::vector<float> values;
        size_t used_tokens = 0;
    };
    
    PagedKVCache(size_t hidden_dim) : hidden_dim_(hidden_dim) {}
    
    void append_tokens(const float* keys, const float* values, size_t num_tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t tokens_remaining = num_tokens;
        size_t offset = 0;
        
        while (tokens_remaining > 0) {
            if (pages_.empty() || pages_.back().used_tokens == PAGE_SIZE) {
                // Allocate new page
                Page new_page;
                new_page.keys.resize(PAGE_SIZE * hidden_dim_);
                new_page.values.resize(PAGE_SIZE * hidden_dim_);
                pages_.push_back(std::move(new_page));
            }
            
            Page& current_page = pages_.back();
            size_t available = PAGE_SIZE - current_page.used_tokens;
            size_t to_copy = std::min(available, tokens_remaining);
            
            // Copy to current page
            std::memcpy(
                current_page.keys.data() + current_page.used_tokens * hidden_dim_,
                keys + offset * hidden_dim_,
                to_copy * hidden_dim_ * sizeof(float)
            );
            std::memcpy(
                current_page.values.data() + current_page.used_tokens * hidden_dim_,
                values + offset * hidden_dim_,
                to_copy * hidden_dim_ * sizeof(float)
            );
            
            current_page.used_tokens += to_copy;
            tokens_remaining -= to_copy;
            offset += to_copy;
        }
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        pages_.clear();
    }
    
    size_t get_total_tokens() const {
        size_t total = 0;
        for (const auto& page : pages_) {
            total += page.used_tokens;
        }
        return total;
    }
    
private:
    size_t hidden_dim_;
    std::vector<Page> pages_;
    std::mutex mutex_;
};

} // namespace Utils
} // namespace CortexAICompression
