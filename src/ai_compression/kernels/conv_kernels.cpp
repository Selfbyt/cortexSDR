/**
 * @file conv_kernels.cpp
 * @brief Implementation of optimized convolution kernels
 */

#include "conv_kernels.hpp"
#include "blas_kernels.hpp"
#include <cstring>
#include <algorithm>

namespace CortexAICompression {
namespace Kernels {

void im2col(const float* data_im,
            int channels, int height, int width,
            int kernel_h, int kernel_w,
            int pad_h, int pad_w,
            int stride_h, int stride_w,
            float* data_col) {
    const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    const int channel_size = height * width;
    
    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_row = -pad_h + kh;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (input_row < 0 || input_row >= height) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad_w + kw;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (input_col >= 0 && input_col < width) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
                data_im += channel_size;
            }
        }
    }
}

void conv2d_im2col(const float* input,
                   const float* weights,
                   const float* bias,
                   float* output,
                   int batch_size,
                   int in_channels, int in_height, int in_width,
                   int out_channels,
                   int kernel_h, int kernel_w,
                   int stride_h, int stride_w,
                   int pad_h, int pad_w) {
    const int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    const int col_rows = in_channels * kernel_h * kernel_w;
    const int col_cols = out_height * out_width;
    
    // Allocate temporary buffer for im2col transformation
    std::vector<float> col_buffer(col_rows * col_cols);
    
    const int input_size = in_channels * in_height * in_width;
    const int output_size = out_channels * out_height * out_width;
    
    for (int b = 0; b < batch_size; ++b) {
        // Transform input to column matrix
        im2col(input + b * input_size,
               in_channels, in_height, in_width,
               kernel_h, kernel_w,
               pad_h, pad_w,
               stride_h, stride_w,
               col_buffer.data());
        
        // Perform GEMM: output = weights * col_buffer
        // weights: (out_channels, col_rows)
        // col_buffer: (col_rows, col_cols)
        // output: (out_channels, col_cols)
        gemm(weights, col_buffer.data(), output + b * output_size,
             out_channels, col_cols, col_rows,
             1.0f, 0.0f, false, false);
        
        // Add bias if provided
        if (bias != nullptr) {
            for (int oc = 0; oc < out_channels; ++oc) {
                float b_val = bias[oc];
                float* out_ptr = output + b * output_size + oc * (out_height * out_width);
                for (int i = 0; i < out_height * out_width; ++i) {
                    out_ptr[i] += b_val;
                }
            }
        }
    }
}

void depthwise_conv2d(const float* input,
                      const float* weights,
                      const float* bias,
                      float* output,
                      int batch_size,
                      int channels,
                      int in_height, int in_width,
                      int kernel_h, int kernel_w,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w) {
    const int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* input_channel = input + (b * channels + c) * in_height * in_width;
            float* output_channel = output + (b * channels + c) * out_height * out_width;
            const float* weight_channel = weights + c * kernel_h * kernel_w;
            
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;
                            
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                sum += input_channel[ih * in_width + iw] * weight_channel[kh * kernel_w + kw];
                            }
                        }
                    }
                    
                    if (bias != nullptr) {
                        sum += bias[c];
                    }
                    
                    output_channel[oh * out_width + ow] = sum;
                }
            }
        }
    }
}

} // namespace Kernels
} // namespace CortexAICompression
