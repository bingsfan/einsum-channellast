在CPU-OPtest的基础上添加一些算子对Channelast格式的支持

test_einsum.cpp中新增了test_einsum4x3channellast  
将输入的真实形状先求出来，即补得0都不要，因为这样可能会影响输出的形状，导致两个输入匹配不上  
比如3x4-2测试项，因此在读文件的时候新增了testutil.h中的readFileChannelast函数  
再在testutil.h中加入对齐的函数，因为gather也用到了  