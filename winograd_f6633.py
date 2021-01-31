import numpy as np
import math

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
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, -1, 2, -2, 1/2, -1/2, 0],
        [0, 1, 1, 4, 4, 1/4, 1/4, 0],
        [0, 1, -1, 8, -8, 1/8, -1/8, 0],
        [0, 1, 1, 16, 16, 1/16, 1/16, 0],
        [0, 1, -1, 32, -32, 1/32, -1/32, 1]
    ]

A = np.array(AT).T

G = [
    [1,     0,      0],
    [-2/9, -2/9, -2/9],
    [-2/9, 2/9, -2/9],
    [1/90, 1/45, 2/45],
    [1/90, -1/45, 2/45],
    [32/45, 16/45, 8/45],
    [32/45, -16/45, 8/45],
    [0, 0, 1]
]

GT = np.array(G).T

BT = [
[1,0,-21/4,0,21/4,0,-1,0],
[0,1,1,-17/4,-17/4,1,1,0],
[0,-1,1,17/4,-17/4,-1,1,0],
[0,1/2,1/4,-5/2,-5/4,2,1,0],
[0,-1/2,1/4,5/2,-5/4,-2,1,0],
[0,2,4,-5/2,-5,1/2,1,0],
[0,-2,4,5/2,-5,-1/2,1,0],
[0,-1,0,21/4,0,-21/4,0,1],
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

def compute_f6633(d, g):
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

def compute_f6633_dims(data, kernel):

    assert len(np.shape(data)) == 4
    assert len(np.shape(kernel)) == 4

    inN, inC, inH, inW = np.shape(data)
    outC, inC, kernelH, kernelW = np.shape(kernel)

    # kernel = np.array([i for i in range(outC * inC * kernelH * kernelW)]).reshape([outC, inC, kernelH, kernelW]).astype('float32')
    # data = np.array([i for i in range(inC * inH * inW)]).reshape([1, inC, inH, inW]).astype('float32')

    final_result = []
    for oc in range(outC):
        kernel_c = kernel[oc]
        result_c = 0
        for ic in range(inC):
            cur_data = data[0, ic, :, :]
            cur_kernel = kernel_c[ic]
            out_c = compute_f6633(cur_data, cur_kernel)
            result_c += out_c
        final_result.append(result_c)
    # print(final_result)

    return final_result


def conv_winograd_6633(data, kernel):
    assert len(np.shape(data)) == 4
    assert len(np.shape(kernel)) == 4

    inN, inC, inH, inW = np.shape(data)
    outC, inC, kernelH, kernelW = np.shape(kernel)
    outH = inH - kernelH + 1
    outW = inW - kernelW + 1

    print(inN, inC, inH, inW, outC, kernelH, kernelW)

    m = 6
    r = 3
    tailSize = m + r - 1

    inHPad = math.ceil(outH / m) * m + r -1
    inWPad = math.ceil(outW / m) * m + r -1

    # tail个数
    nbTaiY = math.floor(inHPad / 6)
    nbTaiX = math.floor(inWPad / 6)

    #tail 输入大小
    outHTM = nbTaiY * m
    outWTM = nbTaiX * m


    # step1:对输入做pad
    dataPad = np.zeros((inN, inC, inHPad, inWPad))
    dataPad[:, :, :inH, :inW] = data
    print(dataPad)
    print(np.shape(dataPad))

    #step2:切分tail
    outTM = np.zeros((1, outC, outHTM, outWTM))

    print("tail x y : %d %d" % (nbTaiX, nbTaiY))
    for j in range(nbTaiY):
        for i in range(nbTaiX):
            hIdxStart = j * m
            wIdxStart = i * m

            # 重叠
            # if hIdxStart > 0:
            #     hIdxStart -= (r - 1)
            # if wIdxStart > 0:
            #     wIdxStart -= (r - 1)

            hIdxEnd = hIdxStart +  tailSize
            wIdxEnd = wIdxStart +  tailSize


            tail = dataPad[:, :, hIdxStart:hIdxEnd, wIdxStart:wIdxEnd]
            print(i, hIdxStart, hIdxEnd, wIdxStart, wIdxEnd)
            # print(np.shape(tail))
            tailOut = compute_f6633_dims(tail, kernel)

            outHIdxStart = m * j
            outHIdxEnd = m * (j + 1)
            outWIdxStart = m * i
            outWIdxEnd = m * (i + 1)
            outTM[:, :, outHIdxStart:outHIdxEnd, outWIdxStart:outWIdxEnd] = tailOut

    # print(outTM)
    # print(np.shape(outTM))   
    out = outTM[:, :, :outH, :outW]         
    print(out)
    print(np.shape(out))            

def unit_test_winograd_6644():
    inC = 3
    inH = 8
    inW = 8
    outC = 32
    kernelH = 3
    kernelW = 3
    compute_f6633_dims(inC, inH, inW, outC, kernelH, kernelW)


def main():
    inC = 3
    inH = 40
    inW = 40
    outC = 32
    kernelH = 3
    kernelW = 3

    kernel = np.array([i for i in range(outC * inC * kernelH * kernelW)]).reshape([outC, inC, kernelH, kernelW]).astype('float32')
    data = np.array([i for i in range(inC * inH * inW)]).reshape([1, inC, inH, inW]).astype('float32')

    conv_winograd_6633(data, kernel)


if __name__ == '__main__':
    main()
    