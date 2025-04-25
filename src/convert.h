#ifndef COVERT.H
#define COVERT_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "mat.h"

using namespace std;

// 从文本文件读取浮点数到向量
static std::vector<float> readFloatFile(const std::string &filename) {
	std::ifstream infile(filename);
	if(!infile)
		throw std::runtime_error("无法打开输入文件: " + filename);
	std::vector<float> data;
	float v;
	while(infile >> v)
		data.push_back(v);
	return data;
}

// 将浮点数向量写入文本文件，每行一个数
static void writeFloatFile(const std::string &filename, const std::vector<float> &data) {
	std::ofstream outfile(filename);
	if(!outfile)
		throw std::runtime_error("无法打开输出文件: " + filename);
	outfile << std::fixed << std::setprecision(6);
	for(float v : data)
		outfile << v << '\n';
}


/* -----------------------------channellast_to_chw------------------------------------- */

static float *channellast_to_chw(float *input_data, size_t H, size_t W, size_t C, size_t realc,
								 size_t align) {
	if(C % align != 0) {
		cerr << "通道数 " << C << " 不是 " << align << " 的整数倍，数据格式错误！" << endl;
		return nullptr;
	}
	size_t block_num = C / align;

	// 只保留 realc 个通道
	float *chw_data = (float *)malloc(realc * H * W * sizeof(float));
	if(!chw_data) {
		cerr << "内存分配失败！" << endl;
		return nullptr;
	}

	// 输入: (block_num, H, W, align)
	// 输出: (realc, H, W)
	for(size_t c = 0; c < realc; ++c) {
		size_t b			   = c / align;
		size_t offset_in_block = c % align;

		for(size_t h = 0; h < H; ++h) {
			for(size_t w = 0; w < W; ++w) {
				size_t input_index =
				 b * (H * W * align) + h * (W * align) + w * align + offset_in_block;

				size_t output_index = c * (H * W) + h * W + w;

				chw_data[output_index] = input_data[input_index];
			}
		}
	}
	return chw_data;
}

static float *channellast_to_nchw(float *input_data, size_t N, size_t H, size_t W,
										  size_t C, size_t realc, size_t align) {
	// auto input_data = readFloatFile(in_file);
	if(C % align != 0) {
		cerr << "通道数 " << C << " 不是 " << align << " 的整数倍，数据格式错误！" << endl;
		return nullptr;
	}
	size_t block_num = C / align;

	// 最终输出: NCHW，且只保留 realc 个通道
	float *nchw_data = (float *)malloc(N * realc * H * W * sizeof(float));
	if(!nchw_data) {
		cerr << "内存分配失败！" << endl;
		return nullptr;
	}

	// 输入数据布局: (block_num, N, H, W, align)
	// 中间逻辑等价于 NHWC: (N, H, W, C)
	// 最终输出: (N, realc, H, W)
	for(size_t n = 0; n < N; ++n) {
		for(size_t h = 0; h < H; ++h) {
			for(size_t w = 0; w < W; ++w) {
				for(size_t c = 0; c < realc; ++c) {
					size_t b			   = c / align;
					size_t offset_in_block = c % align;

					size_t input_index = b * (N * H * W * align) + n * (H * W * align)
					 + h * (W * align) + w * align + offset_in_block;

					size_t output_index = n * (realc * H * W) + c * (H * W) + h * W + w;

					nchw_data[output_index] = input_data[input_index];
				}
			}
		}
	}
	return nchw_data;
}

/* ------------------------------------------------------------------ */
static void matToFloatArray4d(float *output, const ncnn::Mat &m) {
	if(output == nullptr) {
		printf("output pointer is null\n");
		return;
	}

	int index = 0;
	for(int q = 0; q < m.c; q++) {
		const float *ptr = m.channel(q);
		for(int z = 0; z < m.d; z++) {
			for(int y = 0; y < m.h; y++) {
				for(int x = 0; x < m.w; x++) {
					output[index++] = ptr[x];
				}
				ptr += m.w;
			}
		}
	}

	// printf("Data copied to float array\n");
}
static void chw_to_channellast(float* output, const char *output_path, int n,
							   int c, int h, int w, int align_to) {
	// 验证输入数据尺寸
	// printf("输入数据尺寸: %d %d %d %d\n", n, c, h, w);
	// printf("输出数据尺寸: %zu\n", output.size());

	// if(output.size() != static_cast<size_t>(n * c * h * w)) {
	// 	throw invalid_argument("输入数据尺寸与指定的n*c*h*w不匹配");
	// }
	// 计算对齐参数
	int padding_channels = (align_to - (c % align_to)) % align_to;
	int total_channels	 = c + padding_channels;

	// 创建并初始化补零后的缓冲区
	vector<float> padded_data(n * total_channels * h * w, 0.0f);

	// 将原始数据拷贝到补零缓冲区
	for(int ni = 0; ni < n; ++ni) {
		for(int ci = 0; ci < c; ++ci) {
			for(int hi = 0; hi < h; ++hi) {
				for(int wi = 0; wi < w; ++wi) {
					int src_idx = ni * c * h * w + ci * h * w + hi * w + wi;
					int dst_idx = ni * total_channels * h * w + ci * h * w + hi * w + wi;
					padded_data[dst_idx] = output[src_idx];
				}
			}
		}
	}

	// 准备结果数据缓冲区
	vector<float> result;
	result.reserve(n * h * w * total_channels);

	// 按对齐分组提取数据
	int num_groups = total_channels / align_to;
	for(int g = 0; g < num_groups; ++g) {
		int channel_start = g * align_to;

		for(int ni = 0; ni < n; ++ni) {
			for(int hi = 0; hi < h; ++hi) {
				for(int wi = 0; wi < w; ++wi) {
					for(int cg = 0; cg < align_to; ++cg) {
						int channel = channel_start + cg;
						int idx = ni * total_channels * h * w + channel * h * w + hi * w + wi;
						result.push_back(padded_data[idx]);
					}
				}
			}
		}
	}
	writeFloatFile(output_path, result);
}



#endif
