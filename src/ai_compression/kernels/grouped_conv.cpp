/**
 * @file grouped_conv.cpp
 * @brief Implementation of grouped convolution kernels
 */

#include "grouped_conv.hpp"
#include "blas_kernels.hpp"
#include <cstring>
#include <algorithm>

namespace CortexAICompression {
namespace Kernels {

void conv2d_grouped(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch,
    size_t in_channels,
    size_t in_height,
    size_t in_width,
    size_t out_channels,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t groups
) {
    // Compute output dimensions
    size_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Channels per group
    size_t in_channels_per_group = in_channels / groups;
    size_t out_channels_per_group = out_channels / groups;
    
    // Initialize output
    std::fill_n(output, batch * out_channels * out_height * out_width, 0.0f);
    
    // Process each group independently
    for (size_t g = 0; g < groups; ++g) {
        size_t in_channel_start = g * in_channels_per_group;
        size_t out_channel_start = g * out_channels_per_group;
        
        // Process each batch
        for (size_t b = 0; b < batch; ++b) {
            // Process each output channel in this group
            for (size_t oc = 0; oc < out_channels_per_group; ++oc) {
                size_t out_ch = out_channel_start + oc;
                
                // Add bias if provided
                if (bias) {
                    for (size_t oh = 0; oh < out_height; ++oh) {
                        for (size_t ow = 0; ow < out_width; ++ow) {
                            size_t out_idx = ((b * out_channels + out_ch) * out_height + oh) * out_width + ow;
                            output[out_idx] = bias[out_ch];
                        }
                    }
                }
                
                // Convolve with each input channel in this group
                for (size_t ic = 0; ic < in_channels_per_group; ++ic) {
                    size_t in_ch = in_channel_start + ic;
                    
                    // Convolve kernel with input
                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            // Weight index
                            size_t weight_idx = (((out_ch * in_channels_per_group + ic) * kernel_h + kh) * kernel_w + kw);
                            float w = weights[weight_idx];
                            
                            // Apply to all output positions
                            for (size_t oh = 0; oh < out_height; ++oh) {
                                for (size_t ow = 0; ow < out_width; ++ow) {
                                    int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
                                    int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                                    
                                    if (ih >= 0 && ih < static_cast<int>(in_height) &&
                                        iw >= 0 && iw < static_cast<int>(in_width)) {
                                        size_t in_idx = ((b * in_channels + in_ch) * in_height + ih) * in_width + iw;
                                        size_t out_idx = ((b * out_channels + out_ch) * out_height + oh) * out_width + ow;
                                        output[out_idx] += input[in_idx] * w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void depthwise_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch,
    size_t channels,
    size_t in_height,
    size_t in_width,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w
) {
    // Compute output dimensions
    size_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Initialize output
    std::fill_n(output, batch * channels * out_height * out_width, 0.0f);
    
    // Process each batch and channel independently
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            // Add bias
            if (bias) {
                for (size_t oh = 0; oh < out_height; ++oh) {
                    for (size_t ow = 0; ow < out_width; ++ow) {
                        size_t out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        output[out_idx] = bias[c];
                    }
                }
            }
            
            // Apply depthwise kernel
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    size_t weight_idx = (c * kernel_h + kh) * kernel_w + kw;
                    float w = weights[weight_idx];
                    
                    for (size_t oh = 0; oh < out_height; ++oh) {
                        for (size_t ow = 0; ow < out_width; ++ow) {
                            int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
                            int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                            
                            if (ih >= 0 && ih < static_cast<int>(in_height) &&
                                iw >= 0 && iw < static_cast<int>(in_width)) {
                                size_t in_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                size_t out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                                output[out_idx] += input[in_idx] * w;
                            }
                        }
                    }
                }
            }
        }
    }
}

void pointwise_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch,
    size_t in_channels,
    size_t out_channels,
    size_t height,
    size_t width
) {
    // 1x1 convolution is just a matrix multiplication per spatial location
    // Reshape: (batch * H * W) x in_channels @ in_channels x out_channels
    //        = (batch * H * W) x out_channels
    
    size_t spatial_size = height * width;
    size_t total_spatial = batch * spatial_size;
    
    // We can use GEMM directly
    // For efficiency, we reshape input to (batch*H*W, in_channels)
    // and multiply with weights^T (out_channels, in_channels)
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < spatial_size; ++s) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                float sum = bias ? bias[oc] : 0.0f;
                
                for (size_t ic = 0; ic < in_channels; ++ic) {
                    size_t in_idx = (b * in_channels + ic) * spatial_size + s;
                    size_t w_idx = oc * in_channels + ic;
                    sum += input[in_idx] * weights[w_idx];
                }
                
                size_t out_idx = (b * out_channels + oc) * spatial_size + s;
                output[out_idx] = sum;
            }
        }
    }
}

} // namespace Kernels
} // namespace CortexAICompression
