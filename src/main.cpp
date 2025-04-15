#include "mytestutil.h"
#include "testutil.h"
#include <iostream>
#include <list>
#include <stack>
#include <sys/time.h>
#include <functional>
#include <unordered_map>
#include <vector>
using namespace std;

// 性能测试函数
void time_test_einsum(const string& test_name, function<void()> test_func) {
    struct timeval start, end;
    const int runs = 20; // 运行20次取平均值

    gettimeofday(&start, NULL);
    for(int i = 0; i < runs; i++) {
        test_func();
    }
    gettimeofday(&end, NULL);

    float time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    float avg_time = time_use / 1000 / runs; // 转换为毫秒并计算平均值

    cout << test_name << " average time: " << avg_time << " ms (over " << runs << " runs)"
         << endl;
}

int main(int argc, char* argv[]) {
#if 1
    // 4x3-1 测试
    time_test_einsum("Einsum 4x3-1 channellast test", []() {
        test_einsum4x3channellast(
            {2, 14, 80, 16}, 
            {14, 80, 16},
            "input/einsum_4_3_1/data1-channellast16.txt",
            "input/einsum_4_3_1/data2-channellast16.txt", 
            "ijml,kml->ijkl",
            "input/einsum_4_3_1/output-mychannellast16.txt", 
            16);
    });
#endif

#if 0
    // 4x3-2 测试
    time_test_einsum("Einsum 4x3-2 channellast test", []() {
        test_einsum4x3channellast(
            {2, 14, 80, 64},
            {14, 80, 64},
            "input/einsum_4_3_2/data1-channellast64.txt",
            "input/einsum_4_3_2/data2-channellast64.txt",
            "ijml,kmj->ijkl",
            "input/einsum_4_3_2/output-mychannellast64.txt",
            64);
    });
#endif

#if 1
    // 4x4-1 测试
    time_test_einsum("Einsum 4x4-1 channellast test", []() {
        test_einsum4x4channnellast( 
            {2, 16, 80, 16},
            {2, 16, 80, 16},
            "input/einsum_4_4_1/data1-channellast16.txt",
            "input/einsum_4_4_1/data2-channellast16.txt",
            "ijml,ikml->ijkl",
            "input/einsum_4_4_1/output-mychannellast16.txt",
            16);
    });
#endif

#if 1
    // 4x4-2 测试
    time_test_einsum("Einsum 4x4-2 channellast test", []() {
        test_einsum4x4channnellast(
            {2, 16, 16, 16},
            {2, 16, 80, 16},
            "input/einsum_4_4_2/data1-channellast16.txt",
            "input/einsum_4_4_2/data2-channellast16.txt",
            "ijml,imkl->ijkl",
            "input/einsum_4_4_2/output-mychannellast16.txt",
            16);
    });
#endif

#if 0
    // 3x4 测试
    time_test_einsum("Einsum 3x4 channellast test", []() {
        test_einsum3x4channnellast(
            {10, 128, 64},
            {1, 16, 16, 128},
            "input/einsum_3_4/mydata1-channellast64.txt",
            "input/einsum_3_4/mydata2-channellast64.txt",
            "lmi,ijkm->ijkl",
            "input/einsum_3_4/myoutput-mychannellast64.txt",
            64);
    });
#endif

#if 0
    // 3x4 小规模测试
    time_test_einsum("Einsum 3x4 small test", []() {
        test_einsum3x4channnellast(
            {5, 6, 2},
            {1, 2, 2, 6},
            "input/einsum_3_4/test_data1-channellast2.txt",
            "input/einsum_3_4/test_data2-channellast2.txt",
            "lmi,ijkm->ijkl",
            "input/einsum_3_4/mytest_output-channellast2.txt",
            2);
    });
#endif

    return 0;
}