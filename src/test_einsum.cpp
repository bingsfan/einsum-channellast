// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// #include "../src/layer/einsum.h"
#include "convert.h"
#include "mytestutil.h"
#include "op.h"
#include "testutil.h"
#include <chrono>
#include <iostream>
#include <string>
using namespace std;

int test_einsum1x1(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path) {
	printf("=====   test_einsum1x1 start   ======\n");
	int ret;
	int size1 = shape1[0];
	int size2 = shape2[0];
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	float *data1 = readFile(file_path1, size1);
	float *data2 = readFile(file_path2, size2);
	data[0]		 = InitMat1D_float(size1, data1);
	data[1]		 = InitMat1D_float(size2, data2);
	ret			 = einsum(res, data, equation);
	matToFile(output_path, res[0]);

	printf("====    test_einsum1x1 end    =======\n\n");
	return 0;
}
int test_einsum4x3(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path) {
	printf("=====   test_einsum4x3 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
	int w2 = shape2[2], h2 = shape2[1], c2 = shape2[0];
	int input_size1 = w1 * h1 * c1 * b1;
	int input_size2 = w2 * h2 * c2;

	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);

	data[0] = InitMat4D_float(w1, h1, c1, b1, data1);
	data[1] = InitMat3D_float(w2, h2, c2, data2);
	ret		= einsum(res, data, equation);
	matToFIle4d(output_path, res[0]);

	printf("====    test_einsum4x3 end    =======\n\n");
	return 0;
}
int test_einsum4x3channellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
							  const char *file_path2, string equation, const char *output_path,
							  int align_to) {	 // 统一先找到真实形状，再补零
	// printf("=====   test_einsum4x3 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res_origin;
	std::vector<ncnn::Mat> res2;
	int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
	int w2 = shape2[2], h2 = shape2[1], c2 = shape2[0];
	int input_size1 = w1 * h1 * c1 * b1;
	int input_size2 = w2 * h2 * c2;

	int block_size1 = input_size1 / shape1.back() * align_to;
	int block_size2 = input_size2 / shape2.back() * align_to;
	int blockNums	= input_size2 / block_size2;
	vector<vector<float>> vec_data1(blockNums);
	vector<vector<float>> vec_data2(blockNums);

	auto start1	 = chrono::high_resolution_clock::now();
	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);
	auto end1	 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration1 = end1 - start1;
	convert_time_accumulator["read_block_time"] += duration1.count();

	for(int i = 0; i < blockNums; i++) {
		vec_data1[i] = vector<float>(data1 + i * block_size1, data1 + (i + 1) * block_size1);
		vec_data2[i] = vector<float>(data2 + i * block_size2, data2 + (i + 1) * block_size2);
	}

	for(int i = 0; i < blockNums; i++) {
		data[0] = InitMat4D_float(align_to, h1, c1, b1, vec_data1[i].data());
		data[1] = InitMat3D_float(align_to, h2, c2, vec_data2[i].data());
		// 这里一定要使用深拷贝，浅拷贝会被覆盖
		std::vector<ncnn::Mat> res;
		einsum(res, data, equation);
		res2.push_back(res[0]);
	}
	auto start2 = chrono::high_resolution_clock::now();
	writeAllMats4dToFile(output_path, res2);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_block_time"] += duration2.count();
	// printf("====    test_einsum4x3 end    =======\n\n");
	return 0;
}
int test_einsum3x4(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path) {
	printf("=====   test_einsum3x4 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
	int input_size1 = w1 * h1 * c1;
	int input_size2 = w2 * h2 * c2 * b2;

	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);

	data[0]	 = InitMat3D_float(w1, h1, c1, data1);
	data[1]	 = InitMat4D_float(w2, h2, c2, b2, data2);
	ret		 = einsum(res, data, equation);
	int resw = res[0].w;
	int resh = res[0].h;
	int resd = res[0].d;
	int resc = res[0].c;
	matToFIle4d(output_path, res[0]);

	printf("====    test_einsum3x4 end    =======\n\n");
	return 0;
}
/*
	这个3x4的太难弄了，弄不出来，不如直接转换一下算了
*/
int test_einsum3x4channnellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
							   const char *file_path2, string equation,
							   const char *output_path, int align_to)

{
	printf("=====   test_einsum3x4 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	std::vector<ncnn::Mat> res2;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
	int input_size1 = w1 * h1 * c1;
	int input_size2 = w2 * h2 * c2 * b2;

	int block_size = input_size2 / shape2.back() * align_to;
	int blockNums  = input_size2 / block_size;
	vector<vector<float>> vec_data1(blockNums);
	vector<vector<float>> vec_data2(blockNums);

	int out_len;
	int realw1 = b2;

	// float *data1 = readFile(file_path1, input_size1);
	float *data1 = readFileChannellast(file_path1, c1, h1, w1, align_to, realw1, out_len);
	vector<float> vec_561(data1, data1 + out_len);
	outputFile_line("input/einsum_3_4/561.txt", vec_561);
	float *data2	 = readFile(file_path2, input_size2);
	float *data1_561 = readFile("input/einsum_3_4/561.txt", out_len);

	for(int i = 0; i < blockNums; i++) {
		vec_data2[i] = vector<float>(data2 + i * block_size, data2 + (i + 1) * block_size);
	}
	int data1_block_size = out_len / blockNums;
	for(int i = 0; i < blockNums; i++) {
		vec_data1[i] =
		 vector<float>(data1 + i * data1_block_size, data1 + (i + 1) * data1_block_size);
	}
	int resw, resh, resd, resc;

	for(int i = 0; i < blockNums; i++) {
		data[0] = myInitMat3D_float(1, 6, 5, vec_561.data());
		data[1] = InitMat4D_float(align_to, h2, c2, b2, vec_data2[i].data());
		print_mat4d(data[0]);
		// print_mat4d(data[1]);
		einsum(res, data, equation);
		/*      //调试使用，看每一块的运算结果对不对
				print4DMatWithIndex("data1", i, data[0]);
				print4DMatWithIndex("data2", i, data[1]);
				print4DMatWithIndex("res", i, res[0]);
				 */
		res2.push_back(res[0]);
	}
	resw = res[0].w;
	resh = res[0].h;
	resd = res[0].d;
	resc = res[0].c;
	sumAndWriteMatToFile(output_path, res2);
	align_channels(output_path, 1, 1, resc, resd, resh, resw, align_to);
	printf("====    test_einsum3x4 end    =======\n\n");
	return 0;
}

int test_einsum4x4(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path) {
	printf("=====   test_einsum4x4 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
	int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
	int input_size1 = w1 * h1 * c1 * b1;
	int input_size2 = w2 * h2 * c2 * b2;

	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);

	data[0] = InitMat4D_float(w1, h1, c1, b1, data1);
	data[1] = InitMat4D_float(w2, h2, c2, b2, data2);
	ret		= einsum(res, data, equation);
	matToFIle4d(output_path, res[0]);

	printf("====    test_einsum4x4 end    =======\n\n");
	return 0;
}
int test_einsum4x4channnellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
							   const char *file_path2, string equation,
							   const char *output_path, int align_to)

{
	// printf("=====   test_einsum4x4 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res_origin;
	std::vector<ncnn::Mat> res2;
	int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
	int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
	int input_size1 = w1 * h1 * c1 * b1;
	int input_size2 = w2 * h2 * c2 * b2;

	int block_size1 = input_size1 / shape1.back() * align_to;
	int block_size2 = input_size2 / shape2.back() * align_to;
	int blockNums	= input_size2 / block_size2;
	vector<vector<float>> vec_data1(blockNums);
	vector<vector<float>> vec_data2(blockNums);

	auto start1	 = chrono::high_resolution_clock::now();
	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);
	auto end1	 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration1 = end1 - start1;
	convert_time_accumulator["read_block_time"] += duration1.count();

	for(int i = 0; i < blockNums; i++) {
		vec_data1[i] = vector<float>(data1 + i * block_size1, data1 + (i + 1) * block_size1);
		vec_data2[i] = vector<float>(data2 + i * block_size2, data2 + (i + 1) * block_size2);
	}
	for(int i = 0; i < blockNums; i++) {
		data[0] = InitMat4D_float(align_to, h1, c1, b1, vec_data1[i].data());
		data[1] = InitMat4D_float(align_to, h2, c2, b2, vec_data2[i].data());
		std::vector<ncnn::Mat> res;
		einsum(res, data, equation);
		res2.push_back(res[0]);
	}
	auto start2 = chrono::high_resolution_clock::now();
	writeAllMats4dToFile(output_path, res2);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_block_time"] += duration2.count();
	// printf("====    test_einsum4x4 end    =======\n\n");
	return 0;
}
// 还要测格式转换的时间
int test_einsum_transfer(vector<int> shape1, vector<int> shape2, const char *file_path1,
						 const char *file_path2, string equation, const char *output_path,
						 int align_to, int realc1, int realc2) {
	// printf("=====   einsum-transfer start   ======\n");
	// 这个就直接从chw转换把，转hwc要改好多东西
	int input_size1 = 1;
	for(auto i : shape1) {
		input_size1 *= i;
	}
	int input_size2 = 1;
	for(auto i : shape2) {
		input_size2 *= i;
	}
	size_t n1, h1, w1, c1;
	size_t n2, h2, w2, c2;
	int data1_dims = shape1.size();
	int data2_dims = shape2.size();
	float *data1_chw;
	float *data2_chw;
	// 在这里读取并且转换成chw的,之后再添加测时间的
	auto start1	 = chrono::high_resolution_clock::now();
	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);
	if(data1_dims == 3 && data2_dims == 4) {
		h1 = shape1[0], w1 = shape1[1], c1 = shape1[2],n1=1;
		n2 = shape2[0], h2 = shape2[1], w2 = shape2[2], c2 = shape2[3];
		data1_chw = channellast_to_chw(data1, h1, w1, c1, realc1, align_to);
		data2_chw = channellast_to_nchw(data2, n2, h2, w2, c2,realc2, align_to);
	} else if(data1_dims == 4 && data2_dims == 3) {
		n1 = shape1[0], h1 = shape1[1], w1 = shape1[2], c1 = shape1[3];
		h2 = shape2[0], w2 = shape2[1], c2 = shape2[2],n2=1;
		data1_chw = channellast_to_nchw(data1, n1, h1, w1, c1, realc1, align_to);
		data2_chw = channellast_to_chw(data2, h2, w2, c2, realc2, align_to);
	} else if(data1_dims == 4 && data2_dims == 4) {
		n1 = shape1[0], h1 = shape1[1], w1 = shape1[2], c1 = shape1[3];
		n2 = shape2[0], h2 = shape2[1], w2 = shape2[2], c2 = shape2[3];
		data1_chw = channellast_to_nchw(data1, n1, h1, w1, c1, realc1, align_to);
		data2_chw = channellast_to_nchw(data2, n2, h2, w2, c2, realc2, align_to);
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration1 = end1 - start1;
	convert_time_accumulator["read_and_convert_block_to_chw"] += duration1.count();

	// 计算einsum
	int ret;
	vector<ncnn::Mat> data(2);
	vector<ncnn::Mat> res_origin(1);
	// 这是计算的过程
	if(data1_dims == 3 && data2_dims == 4) {
		data[0] = InitMat3D_float(w1, h1, realc1, data1_chw);
		data[1] = InitMat4D_float(w2, h2, realc2, n2, data2_chw);
		ret		= einsum(res_origin, data, equation);
	} else if(data1_dims == 4 && data2_dims == 3) {
		data[0] = InitMat4D_float(w1, h1, realc1, n1, data1_chw);
		data[1] = InitMat3D_float(w2, h2, realc2, data2_chw);
		ret		= einsum(res_origin, data, equation);
	} else if(data1_dims == 4 && data2_dims == 4) {
		data[0] = InitMat4D_float(w1, h1, realc1, n1, data1_chw);
		data[1] = InitMat4D_float(w2, h2, realc2, n2, data2_chw);
		ret		= einsum(res_origin, data, equation);
	}
	// 这里是转换成Channelast+写文件的的过程
	// 输出都是四维，直接转换然后输出就行了
	int resw = res_origin[0].w;
	int resh = res_origin[0].h;
	int resd = res_origin[0].d;
	int resc = res_origin[0].c;
	int output_size = resw * resh * resd * resc;
	auto start2		   = std::chrono::high_resolution_clock::now();
	float *result_data = new float[output_size];
	matToFloatArray4d(result_data, res_origin[0]);
	// outputFile_line("einsum_result.txt", vector<float>(result_data, result_data + output_size));
	chw_to_channellast(result_data,output_path,resc,resd,resh,resw,align_to);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_and_convert_block_to_chw"] += duration2.count();
	// printf("====    einsum-transfer end    =======\n\n");
	return 0;
}