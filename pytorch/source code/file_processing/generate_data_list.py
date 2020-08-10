import os

def generate(basedir , imgs, labels):
    # basedir 根目录
    # imgs 根目录下的图片文件夹的名称
    # labels 根目录下的标签文件夹的名称
    img_list = os.listdir(os.path.join(basedir, imgs))
    lb_list = os.listdir(os.path.join(basedir, labels))
    img_list.sort() #由于同名仅仅后缀不相同的关系 故而sort之后必然保持对应关系
    lb_list.sort()
    with open(os.path.join(basedir, 'train.txt'), 'w') as f:
        assert len(img_list) == len(lb_list)
        for i in range(len(img_list)):
            item = os.path.join(imgs, img_list[i]) + ' ' + os.path.join(labels, lb_list[i]) + '\n'
            f.write(item)


if __name__ == "__main__":
    train_list = './dataset/train.txt'
    train_dir = './dataset/'
    generate(train_dir, 'imgs', 'masks')