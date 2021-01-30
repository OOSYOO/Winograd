import numpy as np


def im2col(data, inC, inH, inW, outH, outW, kernelH, kernelW):
    data_im2col = np.ones((inC * kernelH * kernelW * outH*outW))
    # print("out shape : ", np.shape(data_im2col))

    dst_idx = 0
    for c in range(inC):
        for j in range(kernelH):
            for i in range(kernelW):
                for y in range(outH):
                    for x in range(outW):
                        row = y + j
                        col = x + i
                        # print(row, col, data[0, 0, row, col])

                        data_im2col[dst_idx] = data[0, c, row, col]
                        dst_idx += 1

    data_im2col = data_im2col.reshape((inC * kernelH * kernelW, outH*outW))
    return data_im2col

def im2col_kernel(data, outC, inC, kernelH, kernelW):
    data_im2col = np.ones((outC * inC * kernelH * kernelW))
    # print("out shape : ", np.shape(data_im2col))

    dst_idx = 0
    for c in range(outC):
        for j in range(inC):
            for i in range(kernelH):
                for y in range(kernelW):
                    data_im2col[dst_idx] = data[c, j, i, y]
                    dst_idx += 1

    data_im2col = data_im2col.reshape((outC, inC * kernelH * kernelW))
    return data_im2col

def main():
    inC = 1
    inH = 4
    inW = 4
    outC = 1
    outH = 2
    outW = 2 
    kernelH = 3
    kernelW = 3

    #input
    data = np.array([i for i in range(inC * inH * inW)]).reshape([1, inC, inH, inW]).astype('float32')
    print(data)
    # print(np.shape(data))
    # data = np.ones([inC * inH * inW]).reshape([1, inC, inH, inW]).astype('float32')
    data_im2col = im2col(data, inC, inH, inW, outH, outW, kernelH, kernelW)
    # print(data_im2col)
    # print(np.shape(data_im2col))

    #kernel
    kernel = np.array([i for i in range(outC * inC * kernelH * kernelW)]).reshape([outC, inC, kernelH, kernelW]).astype('float32')
    # data = np.ones([inC * inH * inW]).reshape([1, inC, inH, inW]).astype('float32')
    print(kernel)
    # print(np.shape(kernel))
    kernel_im2col = im2col_kernel(kernel, outC, inC, kernelH, kernelW)
    # print(kernel_im2col)
    # print(np.shape(kernel_im2col))

    print(data_im2col)
    print(kernel_im2col)

    out = np.matmul(kernel_im2col, data_im2col)
    print(out)


if __name__ == '__main__':
    main()
    