import onnxruntime
import numpy as np
from onnxruntime.datasets import get_example
from model import segnet
import torch
device = torch.device('cpu')
ckpt_path = './esp_dict.pth'
# 得到torch模型的输出
dummy_input = torch.randn(1, 3, 256, 256, device=device)

myModel = segnet.SegMattingNet()
state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
myModel.load_state_dict(state_dict)
myModel.eval()

with torch.no_grad():
    _, torch_out = myModel(dummy_input)
print(torch_out)

# 得到onnx模型的输出
example_model = get_example('D:\code\pycharm\checkonnx\esp_newtrain.onnx') #一定要写绝对路径
sess = onnxruntime.InferenceSession(example_model)
onnx_out = sess.run(None, {"input":dummy_input.data.numpy()})

# 判断输出结果是否一致，小数点后3位一致即可
res = np.testing.assert_almost_equal(torch_out.data.numpy(), onnx_out[0], decimal=3)
print(onnx_out[0])
print(res)