
import numpy as np
import cv2
'''
import torch
a = torch.arange(10)
print(a)
print(a.type())
print(a.shape)
# print(a.size) 打印的是方法信息 built-in method size of tensor object......
print(a.size())
#a_val = a.item() #ValueError: only one element tensors can be converted to Python scalars
a_numpy = a.numpy()
print(a_numpy)
'''

#np.set_printoptions(threshold=np.inf)
mask_lb = cv2.imread('./dataset/labels/1803151818-00000003.png', cv2.IMREAD_UNCHANGED)
print(mask_lb[:, :, 3].shape)
print(mask_lb[:, :, 3].dtype)
mask_lb = mask_lb.astype(np.float32)
mask_lb = mask_lb[:, :, 3]
print(mask_lb)
cv2.imshow('mask_0_255', mask_lb)
cv2.waitKey(0)

mask_01 = mask_lb / 255

print(mask_01)
cv2.imshow('mask_0_1', mask_01)
cv2.waitKey(0)

cv2.imshow('reverse', 1 - mask_01)
cv2.waitKey(0)