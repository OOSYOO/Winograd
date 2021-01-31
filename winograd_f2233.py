import numpy as np

# Y = AT[(GgGT)·(BTdB)]

'''
step:
  1 变换输入
  2 变换卷积核
  3 dot
  4 变换输出
'''


'''
F(2, 3)各个步骤的维度
F(m, r) = m + r -1
    输出个数为m, 参数为r，的FIR滤波器，不需要m*r次乘法，只需要m+r-1次乘法。

输入变换矩阵BT : [m+r-1, m+r-1]
输入矩阵d : [m+r-1, m+r-1]

卷积核变换矩阵G : [m+r-1, r]
卷积核g : [r, r]

输出变换矩阵：AT[m, m+r-1]


Gg : [m+r-1, m+r-1]
BTdB: [m+r-1, m+r-1]
Y : [m, m]
'''


AT = [
        [1,     1,      1,      0],
        [0,     1,      -1,     1]
    ]

A = np.array(AT).T

G = [
    [1,     0,      0],
    [1/2,   1/2,    1/2],
    [1/2,   -1/2,   1/2],
    [0,     0,      1]
]

GT = np.array(G).T

BT = [
    [1,     0,      -1,     0],
    [0,     1,       1,     0],
    [0,     -1,      1,     0],
    [0,     -1,      0,     1]
]

B = np.array(BT).T

def trans_input(BT, d, B):
    return np.linalg.multi_dot([BT, d, B])

def trans_kernel(G, g, GT):
    return np.linalg.multi_dot([G, g, GT])

def trans_output(AT, M, A):
    return np.linalg.multi_dot([AT, M, A])

def print_val_shape(arr, name = ""):
    print('#'*10 + ' '*5 + name  + ' '*5 + '#'*10)
    print(arr)
    print(np.shape(arr))

def compute_f2233(d, g):
    # d = np.array([i for i in range(16)]).reshape((4, 4))
    # g = np.array([i for i in range(9)]).reshape((3, 3))

    V = trans_input(BT, d, B)
    U = trans_kernel(G, g, GT)
    M = V * U
    out = trans_output(AT, M, A)
    

    # print_val_shape(V, "U")
    # print_val_shape(U, "V")
    # print_val_shape(M, "M")
    # print_val_shape(out, "out")

    return out

def compute_f2233_dims():
    inC = 3
    inH = 4
    inW = 4
    outC = 32
    outH = 2
    outW = 2 
    kernelH = 3
    kernelW = 3
    kernel = np.array([i for i in range(outC * inC * kernelH * kernelW)]).reshape([outC, inC, kernelH, kernelW]).astype('float32')
    data = np.array([i for i in range(inC * inH * inW)]).reshape([1, inC, inH, inW]).astype('float32')

    final_result = []
    for oc in range(outC):
        kernel_c = kernel[oc]
        result_c = 0
        for ic in range(inC):
            cur_data = data[0, ic, :, :]
            cur_kernel = kernel_c[ic]
            out_c = compute_f2233(cur_data, cur_kernel)
            result_c += out_c
        final_result.append(result_c)
    print(final_result)

def main():
    # compute_f2233()
    compute_f2233_dims()


if __name__ == '__main__':
    main()
    