import os
import shutil
# usage 文件遍历与拷贝
source_path = r"C:\Users\cz\Desktop\fvc_test"
target_path = r"C:\Users\cz\Desktop\fvc_process"

if not os.path.exists(os.path.join(target_path, "imgs")):
    os.mkdir(os.path.join(target_path, "imgs"))
if not os.path.exists(os.path.join(target_path, "alphas")):
    os.mkdir(os.path.join(target_path, "alphas"))
if not os.path.exists(os.path.join(target_path, "masks")):
    os.mkdir(os.path.join(target_path, "masks"))

for root, dirs, files in os.walk(source_path):
    for file in files:
        # 获取文件路径
        full_path = os.path.join(root, file)
        suffix = os.path.splitext(full_path)[-1]
        if suffix == '.png':
            shutil.copyfile(full_path, os.path.join(target_path, "alphas", file))

            img_path = os.path.splitext(full_path)[0]
            img_path += '.jpg'
            shutil.copyfile(img_path, os.path.join(target_path, "imgs", file))

