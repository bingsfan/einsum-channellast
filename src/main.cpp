#include "mytestutil.h"
#include "testutil.h"
#include <iostream>
#include <list>
#include <stack>
#include <unordered_map>
#include <vector>
using namespace std;

int n;
int m;
vector<vector<int>> directions={{0,1},{1,0},{0,-1},{-1,0}};

int main(int argc, char *argv[]) {
#if 0
    test_einsum4x3(//第一个测试用例
        {2, 14, 80, 16}, 
        {14, 80, 16}, 
        "input/einsum_4_3_1/data1-channellast16.txt", 
        "input/einsum_4_3_1/data2-channellast16.txt", 
        "ijml,kml->ijkl",
        "input/einsum_4_3_1/output-mychannellast16.txt");
#endif
#if 0
	test_einsum4x3channellast({ 2, 14, 80, 16 }, { 14, 80, 16 },
							  "input/einsum_4_3_1/data1-channellast4.txt",
							  "input/einsum_4_3_1/data2-channellast4.txt", "ijml,kml->ijkl",
							  "input/einsum_4_3_1/output-mytest-channellast4.txt", 4);
#endif
#if 0
    test_einsum4x3channellast(
        {2, 14, 80, 16},
        {14, 80, 16},
        "input/einsum_4_3_2/data1-channellast16.txt",
        "input/einsum_4_3_2/data2-channellast16.txt",
        "ijml,kmj->ijkl",
        "input/einsum_4_3_2/output-mychannellast16.txt",16);
#endif
#if 0
    test_einsum4x4( //单通道可以直接用之前的
        {2, 16, 80, 8},
        {2, 16, 80, 8},
        "input/einsum_4_4_1/data1-channellast4.txt",
        "input/einsum_4_4_1/data2-channellast4.txt",
        "ijml,ikml->ijkl",
        "input/einsum_4_4_1/output-mytest-channellast4.txt");
#endif
#if 0
    test_einsum4x4channnellast( //第一个
        {2, 16, 80, 8},
        {2, 16, 80, 8},
        "input/einsum_4_4_1/data1-channellast4.txt",
        "input/einsum_4_4_1/data2-channellast4.txt",
        "ijml,ikml->ijkl",
        "input/einsum_4_4_1/output-mytest-channellast4.txt",
        4);
#endif
#if 0
    test_einsum4x4(
        {2, 16, 16, 16},
        {2, 16, 80, 16},
        "input/einsum_4_4_2/data1-channellast16.txt",
        "input/einsum_4_4_2/data2-channellast16.txt",
        "ijml,imkl->ijkl",
        "input/einsum_4_4_2/output-mychannellast16.txt");
#endif
#if 0
    test_einsum4x4channnellast( //单通道可以直接用之前的,第二个
        {2, 16, 16, 8},
        {2, 16, 80, 8},
        "input/einsum_4_4_2/data1-channellast4.txt",
        "input/einsum_4_4_2/data2-channellast4.txt",
        "ijml,imkl->ijkl",
        "input/einsum_4_4_2/output-mytest-channellast4.txt",
        4);
#endif
#if 0
    test_einsum3x4(
        {100,256,64},
        {1,160,160,256},
        "input/einsum_3_4/data1-channellast64.txt",
        "input/einsum_3_4/data2-channellast64.txt",
        "lmi,ijkm->ijkl",
        "input/einsum_3_4/output-mychannellast64.txt");
#endif

#if 1
    test_einsum3x4channnellast(
        {10, 128, 64},
        {1, 16, 16, 128},
        "input/einsum_3_4/mydata1-channellast64.txt",
        "input/einsum_3_4/mydata2-channellast64.txt",
        "lmi,ijkm->ijkl",
        "input/einsum_3_4/myoutput-mychannellast64.txt",
        64);
#endif
#if 0
    test_einsum3x4channnellast(
        {5, 6, 2},
        {1, 2, 2, 6},
        "input/einsum_3_4/test_data1-channellast2.txt",
        "input/einsum_3_4/test_data2-channellast2.txt",
        "lmi,ijkm->ijkl",
        "input/einsum_3_4/mytest_output-channellast2.txt",
        2);
#endif
}
