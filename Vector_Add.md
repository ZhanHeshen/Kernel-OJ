Vector Add (Basic Template)

向量加法
实现一个加法算子，计算两个 float16 向量的逐元素加法。

C[i] = A[i] + B[i]

函数签名
extern "C" __global__ __aicore__ void add_kernel(
    __gm__ uint8_t* A, __gm__ uint8_t* B, __gm__ uint8_t* C,
    uint32_t totalLength)
示例 1
A = [1.0, 2.0, 3.0, 4.0]
B = [5.0, 6.0, 7.0, 8.0]
C = [6.0, 8.0, 10.0, 12.0]
示例 2
A = [1.5, -2.0, 0.0]
B = [2.5,  4.0, 0.0]
C = [4.0,  2.0, 0.0]
约束
数据类型：float16
1 <= N <= 100000
误差容许：1e-3
提示
模板代码中 Compute() 的 TODO 需要你补全
使用 Add(dst, src1, src2, count) API
