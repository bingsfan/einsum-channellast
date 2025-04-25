# 在CPU-OPtest的基础上添加einsum算子对Channelast格式的支持

只完成了4x3-1,4x4-1和4x4-2  

root@k230:/yolo-test/smh-channellast/einsumTest# ./output/main
Einsum 4x3-1 channellast test average time: 509.569 ms (over 20 runs)
Einsum 4x4-1 channellast test average time: 780.586 ms (over 20 runs)
Einsum 4x4-2 channellast test average time: 788.189 ms (over 20 runs)
