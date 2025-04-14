#ifndef MYTESTUTIL_H
#define MYTESTUTIL_H

#include "mat.h"
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
extern int test_einsum4x3(vector<int> shape1, vector<int> shape2, const char *file_path1, const char *file_path2,
                          string equation, const char *output_path);
extern int test_einsum4x3channellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
                                     const char *file_path2, string equation, const char *output_path, int align_to);
extern int test_einsum3x4channnellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
                                      const char *file_path2, string equation, const char *output_path, int align_to);
extern int test_einsum4x4(vector<int> shape1, vector<int> shape2, const char *file_path1, const char *file_path2,
                          string equation, const char *output_path);
extern int test_einsum3x4(vector<int> shape1, vector<int> shape2, const char *file_path1, const char *file_path2,
                          string equation, const char *output_path);
extern int test_einsum4x4channnellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
                                      const char *file_path2, string equation, const char *output_path, int align_to);
/**
 * @description: 给scatternd三维索引用的，将第一列的数据转移到最后一列去，和input匹配
 * @param {int} *indices
 * @param {int} rows
 * @param {int} cols
 * @return {*}
 */
static void rotateColumns(int *indices, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        std::rotate(indices + i * cols, indices + i * cols + 1, indices + (i + 1) * cols);
    }
}
/**
 * @description: 给scatternd四维索引用的，将第二列的数据转移到最后一列去，和input匹配
 * @param {int} *array
 * @param {int} rows
 * @param {int} cols
 * @return {*}
 */
static void moveSecondColumnToLast(int *array, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        std::rotate(array + i * cols + 1, array + i * cols + 2, array + (i + 1) * cols);
    }
}

/**
 * @description: 带索引的mat打印函数,scatternd调试用的,只能用三维的mat
 * @param {char} *prefix
 * @param {int} index
 * @param {Mat} &m
 * @return {*}
 */
static void printMatWithIndex(const char *prefix, int index, const ncnn::Mat &m)
{
    char filename[256];
    // 生成带索引的文件名，如 input_0.txt, index_0.txt
    snprintf(filename, sizeof(filename), "%s_%d.txt", prefix, index);

    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "无法打开文件 %s\n", filename);
        return;
    }

    // 写入数据
    for (int q = 0; q < m.c; ++q)
    {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; ++y)
        {
            for (int x = 0; x < m.w; ++x)
            {
                fprintf(fp, "%.6f\n", ptr[x]);
            }
            ptr += m.w;
        }
    }

    fclose(fp);
    printf("数据已写入 %s\n", filename);
}
static ncnn::Mat myInitMat3D_float(int w, int h, int c, float *data)
{
    ncnn::Mat m(w, h, c);
    int idx = 0;
    for (int cc = 0; cc < c; cc++)
    {
        float *channel_ptr = m.channel(cc);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                channel_ptr[y * w + x] = data[idx++];
            }
        }
    }
    return m;
}
/**
 * @description: 带索引的mat打印函数,einsum调试用的,只能用四维的mat
 * @param {char} *prefix
 * @param {int} index
 * @param {Mat} &m
 * @return {*}
 */
static void print4DMatWithIndex(const char *prefix, int index, const ncnn::Mat &m)
{
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_%d.txt", prefix, index);

    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "无法打开文件 %s\n", filename);
        return;
    }

    // 四维数据遍历：c -> d -> h -> w
    for (int c = 0; c < m.c; ++c)
    {
        const float *ptr = m.channel(c); // 获取当前通道的起始指针
        for (int d = 0; d < m.d; ++d)
        {
            for (int h = 0; h < m.h; ++h)
            {
                for (int w = 0; w < m.w; ++w)
                {
                    fprintf(fp, "%.6f\n", ptr[w]);
                }
                ptr += m.w; // 移动到下一行（h方向）
            }
        }
    }

    fclose(fp);
    printf("四维数据已写入 %s\n", filename);
}
static void sumAndWriteMatToFile(const char *path, const std::vector<ncnn::Mat> &result)
{
    if (result.empty())
    {
        printf("结果数组为空\n");
        return;
    }

    // 初始化sum_mat为相同形状，填充0
    ncnn::Mat sum_mat = result[0].clone();
    sum_mat.fill(0.0f);

    // 累加所有矩阵
    for (size_t i = 0; i < result.size(); ++i)
    {
        const ncnn::Mat &m = result[i];
        for (int q = 0; q < m.c; q++)
        {
            const float *m_ptr = m.channel(q);
            float *sum_ptr = sum_mat.channel(q);

            // 四维数据遍历：d -> h -> w
            for (int z = 0; z < m.d; z++)
            {
                for (int y = 0; y < m.h; y++)
                {
                    // 累加当前行数据
                    for (int x = 0; x < m.w; x++)
                    {
                        sum_ptr[x] += m_ptr[x];
                    }

                    // 指针移动到下一行
                    m_ptr += m.w;
                    sum_ptr += sum_mat.w; // 保证所有矩阵形状相同
                }
            }
        }
    }

    // 写入文件
    FILE *fp = fopen(path, "w");
    if (fp == nullptr)
    {
        printf("无法打开文件 %s\n", path);
        return;
    }

    // 按通道->深度->高度->宽度的顺序写入
    for (int q = 0; q < sum_mat.c; q++)
    {
        const float *ptr = sum_mat.channel(q);
        for (int z = 0; z < sum_mat.d; z++)
        {
            for (int y = 0; y < sum_mat.h; y++)
            {
                for (int x = 0; x < sum_mat.w; x++)
                {
                    fprintf(fp, "%.6f\n", ptr[x]);
                }
                ptr += sum_mat.w; // 移动到下一行
            }
        }
    }

    fclose(fp);
    printf("求和后的数据已写入 %s\n", path);
}
static void writeAllMats4dToFile(const char *path, const std::vector<ncnn::Mat> &result)
{
    FILE *fp = fopen(path, "w");
    if (fp == nullptr)
    {
        printf("无法打开文件 %s\n", path);
        return;
    }

    for (size_t i = 0; i < result.size(); i++)
    {
        const ncnn::Mat &m = result[i];
        for (int q = 0; q < m.c; q++)
        {
            const float *ptr = m.channel(q);
            for (int z = 0; z < m.d; z++) // 新增d维度循环
            {
                for (int y = 0; y < m.h; y++)
                {
                    for (int x = 0; x < m.w; x++)
                    {
                        fprintf(fp, "%.6f\n", ptr[x]);
                    }
                    ptr += m.w; // 按行步进指针
                }
            }
        }
    }

    fclose(fp);
    printf("所有数据已写入 %s\n", path);
}
static void writeAllMatsToFile(const char *path, const std::vector<ncnn::Mat> &result)
{                                // smh
    FILE *fp = fopen(path, "w"); // 只打开一次文件
    if (fp == nullptr)
    {
        printf("无法打开文件 %s\n", path);
        return;
    }

    // 遍历所有 result 元素
    for (size_t i = 0; i < result.size(); i++)
    {
        const ncnn::Mat &m = result[i];

        // 添加区块分隔标记

        // 写入数据
        for (int q = 0; q < m.c; q++)
        {
            const float *ptr = m.channel(q);
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    fprintf(fp, "%.6f\n", ptr[x]);
                }
                // fprintf(fp, "\n"); // 行尾换行
                ptr += m.w;
            }
            // fprintf(fp, "\n"); // 通道间空行
        }
    }

    fclose(fp);
    printf("所有数据已写入 %s\n", path);
}
// 修改后的文件写入函数（添加索引后缀）
static void matToFileWithIndex(const char *base_path, int index, const ncnn::Mat &m)
{
    char path[256];
    // 生成带索引的唯一文件名，例如 output_0.txt, output_1.txt
    snprintf(path, sizeof(path), "%s_%d.txt", base_path, index);

    FILE *fp = fopen(path, "w");
    if (fp == nullptr)
    {
        printf("无法打开文件 %s\n", path);
        return;
    }

    // 写入维度信息（可选）
    // fprintf(fp, "Channels: %d\nHeight: %d\nWidth: %d\n", m.c, m.h, m.w);

    // 写入数据
    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                fprintf(fp, "%.6f\n", ptr[x]); // 用空格分隔数据点
            }
            ptr += m.w;
        }
    }

    fclose(fp);
    printf("数据已写入 %s\n", path);
}
#endif