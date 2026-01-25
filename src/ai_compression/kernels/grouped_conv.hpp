/**
 * @file grouped_conv.hpp
 * @brief Grouped convolution kernels for efficient depthwise and group convolutions
 */

#pragma once

#include <cstddef>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief Grouped 2D convolution
 * 
 * Standard conv: all input channels connected to all output channels
 * Grouped conv: input/output channels divided into groups, each processed independently
 * 
 * @param input Input tensor (batch x in_channels x height x width)
 * @param weights Weight tensor (out_channels x (in_channels/groups) x kH x kW)
 * @param bias Bias vector (out_channels), can be nullptr
 * @param output Output tensor (batch x out_channels x out_height x out_width)
 * @param batch Batch size
 * @param in_channels Number of input channels
 * @param in_height Input height
 * @param in_width Input width
 * @param out_channels Number of output channels
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param pad_h Padding height
 * @param pad_w Padding width
 * @param groups Number of groups (1 = standard conv, in_channels = depthwise)
 */
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
    size_t groups = 1
);

/**
 * @brief Depthwise separable convolution (groups = in_channels = out_channels)
 * 
 * Optimized path for depthwise convolutions used in MobileNet-style architectures
 * Each input channel is convolved separately with its own kernel
 * 
 * @param input Input tensor
 * @param weights Depthwise kernels (channels x 1 x kH x kW)
 * @param bias Bias vector
 * @param output Output tensor
 * @param batch Batch size
 * @param channels Number of channels
 * @param in_height Input height
 * @param in_width Input width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param pad_h Padding height
 * @param pad_w Padding width
 */
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
);

/**
 * @brief Pointwise (1x1) convolution - optimized GEMM implementation
 * 
 * 1x1 convolutions are pure matrix multiplications and can use optimized BLAS
 * 
 * @param input Input tensor (batch x in_channels x height x width)
 * @param weights Weight tensor (out_channels x in_channels x 1 x 1)
 * @param bias Bias vector (out_channels)
 * @param output Output tensor (batch x out_channels x height x width)
 * @param batch Batch size
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param height Spatial height
 * @param width Spatial width
 */
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
);

} // namespace Kernels
} // namespace CortexAICompression
