#include <iostream>
#include "testutil.h"
/*
    test_einsum.cpp中新增了test_einsum4x3channellast
    将输入的真实形状先求出来，即补得0都不要，因为这样可能会影响输出的形状，导致两个输入匹配不上
    比如3x4-2测试项，因此在读文件的时候新增了testutil.h中的readFileChannelast函数
    再在testutil.h中加入对齐的函数，因为gather也用到了
*/
int main(int argc, char *argv[])
{
    std::cout << "Hello world!" << std::endl;
#if 0
    test_einsum4x3(
        {2, 14, 80, 64},
        {14, 80, 64},
        "input/einsum_4_3_1/data1-channellast64.txt",
        "input/einsum_4_3_1/data2-channellast64.txt",
        "ijml,kml->ijkl",
        "input/einsum_4_3_1/output-mychannellast64.txt");
#endif
#if 1
    test_einsum4x3channellast(
        {2, 14, 80, 64},
        {14, 80, 64},
        "input/einsum_4_3_2/data1-channellast64.txt",
        "input/einsum_4_3_2/data2-channellast64.txt",
        "ijml,kmj->ijkl",
        "input/einsum_4_3_2/output-mychannellast64.txt");
#endif
}
