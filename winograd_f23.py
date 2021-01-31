import numpy as np

# Y = AT[(Gg)·(BTd)]

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
输入矩阵d : [m+r-1, 1]

卷积核变换矩阵G : [m+r-1, r]
卷积核g : [r, 1]

输出变换矩阵：AT[m, m+r-1]


Gg : [m+r-1, 1]
BTd: [m+r-1, 1]
Y : [m, 1]
'''

def trans_input(BT, d):
    return np.matmul(BT, d)

def trans_kernel(G, g):
    return np.matmul(G, g)

def trans_output(AT, M):
    return np.matmul(AT, M)

def print_val_shape(arr, name = ""):
    print('#'*10 + ' '*5 + name  + ' '*5 + '#'*10)
    print(arr)
    print(np.shape(arr))

def compute_f23():
    AT = [
        [1,     1,      1,      0],
        [0,     1,      -1,     1]
    ]

    G = [
        [1,     0,      0],
        [1/2,   1/2,    1/2],
        [1/2,   -1/2,   1/2],
        [0,     0,      1]
    ]

    BT = [
        [1,     0,      -1,     0],
        [0,     1,       1,     0],
        [0,     -1,      1,     0],
        [0,     -1,      0,     1]
    ]

    d = [
        [1],
        [2],
        [3],
        [4],
    ]

    g = [
        [1],
        [2],
        [3],
    ]

    V = trans_input(BT, d)
    U = trans_kernel(G, g)
    M = V * U
    out = trans_output(AT, M)
    

    print_val_shape(V, "U")
    print_val_shape(U, "V")
    print_val_shape(M, "M")
    print_val_shape(out, "out")

def main():
    compute_f23()


if __name__ == '__main__':
    main()
    