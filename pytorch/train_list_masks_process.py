import os
import cv2 as cv
import  numpy as np
def generate_masks_from_matting(SAVE_PATH='./dataset/masks', basedir=r"./dataset/labels"):
    assert os.path.exists(SAVE_PATH)
    imgs = os.listdir(basedir) # 取出文件名字
    idx = 1
    for img in imgs:
        path = os.path.join(basedir, img)
        m = cv.imread(path, cv.IMREAD_UNCHANGED)
        # Otsu 滤波
        ret, mask = cv.threshold(m[:, :, 3], 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
        print(os.path.join(basedir, 'masks', img))
        cv.imwrite(os.path.join(SAVE_PATH, img), mask)
        if idx % 200 == 0:
            print('{} complete.'.format(idx))
        idx+=1


def generate(basedir, imgs, labels):
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


    # test 0-1 mask
    '''
    img = cv.imread(r'./dataset\masks\1803151818-00000003.png', cv.IMREAD_GRAYSCALE) #注意 如果不是灰度图读取的话默认三通道BGR
    print(img.shape)
    cv.imshow('img', img*255)


    img2 = cv.imread(r'./dataset/imgs/1803151818-00000003.jpg')
    cv.imshow('img2', img2)

    img = img[..., np.newaxis]
    fusion = np.multiply(img, img2)
    cv.imshow('fusion', fusion)

    cv.waitKey(0)
    '''
    #generate_masks_from_matting()
    train_list = './dataset/train.txt'
    train_dir = './dataset'
    generate(train_dir, 'imgs', 'masks')