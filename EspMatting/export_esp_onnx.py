"""
Export ONNX model of MODNet with:
    input shape: (batch_size, 3, height, width)
    output shape: (batch_size, 1, height, width)
Arguments:
    --ckpt-path: path of the checkpoint that will be converted
    --output-path: path for saving the ONNX model
Example:
    python export_esp_onnx.py \
        --ckpt-path=modnet_photographic_portrait_matting.ckpt \
        --output-path=modnet_photographic_portrait_matting.onnx
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import segnet


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the ONNX model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    # define model & load checkpoint
    myModel = segnet.SegMattingNet()
    myModel.load_state_dict(torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)['state_dict'])
    myModel.eval()
    myModel.to('cpu')
    # prepare dummy_input
    batch_size = 1
    height = 512
    width = 512
    dummy_input = Variable(torch.randn(batch_size, 3, height, width)).cpu()

    # export to onnx model
    torch.onnx.export(
        myModel, dummy_input, args.output_path, export_params = True,
        input_names = ['input'], output_names = ['output'],
        dynamic_axes = {'input': {0:'batch_size', 2:'height', 3:'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}},
        
        opset_version=11
        )
    import onnxruntime
    print(f'Validating ONNX model.')
    src = torch.randn(1, 3, 512, 512).to('cpu')
    with torch.no_grad():
        alp = myModel(src)
    sess = onnxruntime.InferenceSession(args.output_path)
    out_onnx = sess.run(None, {
            'input': src.cpu().numpy(),
        })
    alp_onnx = torch.as_tensor(out_onnx)
    e = torch.abs(alp.cpu() - alp_onnx).max()
    if e < 0.0005:
        print('Validation passed.')
    else:
        print('Validation failed.')
    




