/**
 * @file conv_kernels.hpp
 * @brief Optimized convolution kernels using im2col + GEMM
 */

#pragma once

#include <vector>
#include <cstddef>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief Im2col transformation for convolution
 * @param data_im Input image in NCHW format
 * @param channels Number of input channels
 * @param height Input height
 * @param width Input width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param pad_h Padding height
 * @param pad_w Padding width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param data_col Output column matrix
 */
void im2col(const float* data_im,
            int channels, int height, int width,
            int kernel_h, int kernel_w,
            int pad_h, int pad_w,
            int stride_h, int stride_w,
            float* data_col);

/**
 * @brief Optimized 2D convolution using im2col + GEMM
 * @param input Input tensor (batch_size, in_channels, height, width)
 * @param weights Filter weights (out_channels, in_channels, kernel_h, kernel_w)
 * @param bias Bias vector (out_channels), can be nullptr
 * @param output Output tensor (batch_size, out_channels, out_height, out_width)
 * @param batch_size Number of samples
 * @param in_channels Input channels
 * @param in_height Input height
 * @param in_width Input width
 * @param out_channels Output channels
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param pad_h Padding height
 * @param pad_w Padding width
 */
void conv2d_im2col(const float* input,
                   const float* weights,
                   const float* bias,
                   float* output,
                   int batch_size,
                   int in_channels, int in_height, int in_width,
                   int out_channels,
                   int kernel_h, int kernel_w,
                   int stride_h, int stride_w,
                   int pad_h, int pad_w);

/**
 * @brief Depthwise separable convolution (more efficient for some architectures)
 * @param input Input tensor
 * @param weights Depthwise weights
 * @param bias Bias vector
 * @param output Output tensor
 * @param batch_size Number of samples
 * @param channels Number of channels (input and output are same)
 * @param in_height Input height
 * @param in_width Input width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param pad_h Padding height
 * @param pad_w Padding width
 */
void depthwise_conv2d(const float* input,
                      const float* weights,
                      const float* bias,
                      float* output,
                      int batch_size,
                      int channels,
                      int in_height, int in_width,
                      int kernel_h, int kernel_w,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w);

} // namespace Kernels
} // namespace CortexAICompression
