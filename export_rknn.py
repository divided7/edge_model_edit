import argparse
import os

from rknn.api import RKNN
import cv2
import numpy as np
import onnxruntime

np.set_printoptions(suppress=True)

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='rope_static_fp16.onnx', help='xxx.onnx')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size, default=1')
    parser.add_argument("--output", type=str, default='wxf_output.rknn', help='xxxx.rknn')
    parser.add_argument("--int8", type=bool, default=False, help='True: int8, False(default): fp16')
    return parser.parse_args()


def build_rknn(opt):
    rknn = RKNN(verbose=True)
    means, stds = [0, 0, 0], [1, 1, 1]
    # means, stds = [[0.485, 0.456, 0.406]], [[0.229, 0.224, 0.225]]
    # means, stds = [[123.675, 116.28, 103.53]], [[58.40, 57.12, 57.38]] 
    print(f"means: {means}\nstds: {stds}")
    rknn.config(target_platform='rk3588', mean_values=means, std_values=stds)
    
    

    
    # load onnx model
    input_size_list = [(1, 3, 544, 960)]
    ret = rknn.load_onnx(model=opt.model, input_size_list=input_size_list)

    if ret != 0:
        print('onnx model load fail!')
        exit(ret)
    else:
        print("\033[91mload onnx model success\033[0m")

    # build
    quant = opt.int8
    print('Do int8 quantization:{}'.format(quant))
    ret = rknn.build(do_quantization=quant, dataset='./dataset.txt', rknn_batch_size=opt.batch_size)
    if ret != 0:
        print('onnx model build fail!')
        exit(ret)
    else:
        print("\033[91mbuild onnx model success\033[0m")

    # export rknn model
    ret = rknn.export_rknn(opt.output)
    if ret != 0:
        print('export rknn failed!')
        exit(ret)
    else:
        print("\033[91mexport rknn success\033[0m")

    # infer
    # 加载rknn模型
    ret = rknn.init_runtime()

    if ret != 0:
        print('Init rknn runtime failed!')
        exit(ret)
    else:
        print("\033[91mInit rknn runtime\033[0m")

    # 加载ONNX模型
    providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(opt.model, providers=providers)

    input_data = np.load("sess_rope_input.npy")
    output_data = np.load("sess_rope_output.npy")

    print("sess_rope_input dtype:")
    print(input_data.dtype, input_data[0, :, 0, 0])

   

    print(f"source outputs: {output_data[0, 0, :15]}")

    # onnx 输入格式为 [b, c, h, w]
    onnx_outputs = ort_session.run(None, {"images_float32": input_data})
    print(f"onnx outputs: {onnx_outputs[0][0, 0, :15]}")


    # 将 NCHW 格式转换为 NHWC 格式  
    input_data_nhwc = np.transpose(input_data, (0, 2, 3, 1))
    print(f"Input data shape: {input_data_nhwc.shape}")

    # rknn 输入格式为 [b, h, w, c]
    rknn_output = rknn.inference(inputs=[input_data_nhwc])
    print(f"rknn outputs: {rknn_output[0][0, 0, :15]}")

     # Print RKNN driver version
    # print(f"RKNN driver version: {rknn.get_sdk_version()}")
  


if __name__ == '__main__':
    opt = opt()
    if opt.int8:
        opt.output = opt.output.replace(".rknn", '_int8.rknn')
    else:
        opt.output = opt.output.replace(".rknn", '_fp16.rknn')
    build_rknn(opt)

# rknn_toolkit 1.3:      conda activate rk
# rknn_toolkit 1.4.6:   source /home/hhd1/project/onnx2rknn/venv_1.4.6/bin/activate
# rknn_toolkit 1.5.2:   source /home/hhd1/project/onnx2rknn/venv_1.5.2/bin/activate
# rknn_toolkit 1.6:      source /home/hhd1/project/onnx2rknn/venv_1.6/bin/activate
