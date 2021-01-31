import sys
import onnx
import onnx.utils
from onnx import helper
from onnx import shape_inference
from onnx import AttributeProto, TensorProto
import numpy as np
from test_onnx import ONNXModel

def list_to_constant(name, shape, data, data_type=None):
    """Generate a constant node using the given infomation.

    :name: the node name and the output value name\\
    :shape: the data shape\\
    :data: the data itself\\
    :returns: the generated onnx constant node
    """
    if not data_type:
        if isinstance(data, int):
            data_type = onnx.helper.TensorProto.INT64
        elif isinstance(data, float):
            data_type = onnx.helper.TensorProto.FLOAT
        elif len(data) > 0 and isinstance(data[0], int):
            data_type = onnx.helper.TensorProto.INT64
        else:
            data_type = onnx.helper.TensorProto.FLOAT
    tensor = onnx.helper.make_tensor(
        name,
        data_type,
        shape,
        data
    )
    new_w_node = onnx.helper.make_node(
        "Constant",
        [],
        [name],
        name = name,
        value = tensor
    )
    return new_w_node


if __name__ == '__main__':
    inC = 3
    inH = 4
    inW = 4
    outC = 32
    outH = 2
    outW = 2 
    kernelH = 3
    kernelW = 3
    kernel = np.array([i for i in range(outC * inC * kernelH * kernelW)])#.reshape([outC, inC, kernelH, kernelW]).astype('float32')
    data = np.array([i for i in range(inC * inH * inW)]).reshape([1, inC, inH, inW]).astype('float32')


    input_tensor_value_info = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,inC,inH,inW])]
    output_tensor_value_info = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,outC,outH,outW])]
    conv1_w = list_to_constant('conv1_weight', [outC, inC, kernelH, kernelW], kernel, TensorProto.FLOAT)


    new_onnx_node_list = []
    new_onnx_node_list.append(conv1_w)

    node_without_padding = onnx.helper.make_node(
        'Conv',
        inputs=['input', 'conv1_weight'],
        outputs=['output'],
        kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[0, 0, 0, 0],
    )
    new_onnx_node_list.append(node_without_padding)


    graph_cnn = helper.make_graph(
        new_onnx_node_list,
        'cnn_test',
        input_tensor_value_info,
        output_tensor_value_info,
    )

    cnn_model = helper.make_model(graph_cnn, producer_name='yue.shen')
    onnx.checker.check_model(cnn_model)

    # save
    onnx.save(cnn_model, "conv.onnx")

    model = ONNXModel("./conv.onnx")
    out = model.forward(data)

    out = np.squeeze(out)
    print(np.shape(out))
    print(out)

    print("done!")