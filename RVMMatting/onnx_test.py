
import onnxruntime as ort
import numpy as np
import cv2
def test_default():
    sess = ort.InferenceSession('rvm_mobilenetv3_fp32.onnx')
    rec = [np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # 必须用模型一样的 dtype
    print(rec)
    downsample_ratio = np.array([0.25], dtype=np.float32)  # 必须是 FP32

    src_img = cv2.imread("jym.jpg")
    src_img = cv2.flip(src_img, 1)
    src = cv2.resize(src_img, (1920, 1080))

    src = src / 255.0 # 归一化
    # print(src.shape)
    # src [h,w,c]
    src = np.transpose(src, (2, 0, 1)).astype(np.float32)# src 张量是 [B, C, H, W] 形状
    src = np.expand_dims(src, 0)
    print(src.shape)

    fgr, pha, *rec = sess.run([], {
        'src': src,
        'r1i': rec[0],
        'r2i': rec[1],
        'r3i': rec[2],
        'r4i': rec[3],
        'downsample_ratio': downsample_ratio
    })

    pha = (pha * 255).astype(np.uint8) #[n,c,h,w]
    pha = np.squeeze(pha, 0)# [c,h,w]
    pha = np.transpose(pha, [1, 2, 0]) #[h, w, c]
    # pha = cv2.repeat(pha, 3)
    pha = cv2.cvtColor(pha, cv2.COLOR_GRAY2BGR)
    pha = cv2.resize(pha, (src_img.shape[1], src_img.shape[0]))
    bgr = cv2.imread('bgr.jpg')
    bgr = cv2.resize(bgr, (src_img.shape[1], src_img.shape[0]))
    com = src_img * (pha/255.0) + (1 - (pha/255.0)) * bgr
    cv2.imwrite("com_jym.png", com)
    cv2.imwrite("pha_jym.png", pha)
    fgr = (fgr * 255).astype(np.uint8)
    #print(fgr.shape)
    fgr = np.squeeze(fgr, 0)
    fgr = np.transpose(fgr, [1, 2, 0])
    # cv2.imshow("pha", pha)
    # cv2.imshow("FGR", fgr)
    # cv2.waitKey(0)

#test_default()
#exit(0)


print('Init WebCam...')
cap = cv2.VideoCapture(0) #创建VideoCapture，传入0即打开系统默认摄像头
W = 1080
H = 720
rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4  # 必须用模型一样的 dtype



print('Start matting...')
import time
sess = ort.InferenceSession('rvm_mobilenetv3_fp32.onnx')
# sess = ort.InferenceSession('rvm_resnet50_fp32.onnx')
rec = [np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # 必须用模型一样的 dtype
# print(rec)
downsample_ratio = np.array([0.2], dtype=np.float32)  # 必须是 FP32
bgr = np.zeros([H, W, 3], np.uint8)
bgr[:, :, 1] = np.zeros([H, W]) + 255


while (True):
    start = time.time()
    _, frame_np = cap.read()
    # frame_np = cv2.flip(frame_np, 1)
    #frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (W, H), cv2.INTER_AREA)
    src = frame_np / 255.0  # 归一化
    # print(src.shape)
    # src [h,w,c]
    src = np.transpose(src, (2, 0, 1)).astype(np.float32)  # src 张量是 [B, C, H, W] 形状
    src = np.expand_dims(src, 0)
    print(src.shape)
    # rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4  # 必须用模型一样的 dtype
    # print(*rec)
    fgr, pha, *rec = sess.run([], {
        'src': src,
        'r1i': rec[0],
        'r2i': rec[1],
        'r3i': rec[2],
        'r4i': rec[3],
        'downsample_ratio': downsample_ratio
    })

    # pha = (pha * 255).astype(np.uint8)  # [n,c,h,w]
    pha = pha.astype(np.uint8)
    pha = np.squeeze(pha, 0)  # [c,h,w]
    pha = np.transpose(pha, [1, 2, 0])  # [h, w, c]

    #cv2.imshow("pha", pha * 255)
    view_np = frame_np * pha + (1 - pha) * bgr
    cv2.imshow("view", view_np)


    #cv2.imshow('rvm - WebCam [Press \'Q\' To Exit]', view_np)
    end = time.time()
    print("fps = ", 1.0/(end - start))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Exit...')