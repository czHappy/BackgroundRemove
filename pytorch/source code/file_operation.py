import os

file_path = './dataset/imgs/1803151818-00000003.jpg'
dir_path = './dataset/imgs'

# 判断文件是否存在

print(os.path.exists(file_path))
print(os.path.exists(dir_path))



# 判断是文件还是目录
print(os.path.isfile(dir_path))
print(os.path.isdir(dir_path))

# 查询dir下的所有文件的名字，包括文件夹的名字

li = os.listdir(dir_path) #Return a list containing the names of the files in the directory.
print(li)

# walk函数 递归遍历树
import shutil
source = r'./dataset/copy_source'
dst_dir = r'./dataset/copy_dst'
def copy_files_to_target(source, dest):

    for root, dirs, files in os.walk(source):
        # print('root = ' , root)
        # print('dir = ' ,dirs)
        # print('files = ' ,files)
        # print('*'*80)
        for file in files:
            src = os.path.join(root, file)
            shutil.copy(src, dest) #os.path.join(dest, file)

copy_files_to_target(source, dst_dir)