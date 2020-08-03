import os

def generate(basedir , imgs, labels):
    img_list = os.listdir(os.path.join(basedir, imgs))
    lb_list = os.listdir(os.path.join(basedir, labels))
    img_list.sort()
    lb_list.sort()
    with open(os.path.join(basedir, 'train.txt'), 'w') as f:
        assert len(img_list) == len(lb_list)
        for i in range(len(img_list)):
            item = os.path.join(imgs, img_list[i]) + ' ' +  os.path.join(labels, lb_list[i]) + '\n'
            f.write(item)



        #for img_name in os.listdir(os.path.join(basedir, prefix)):
        #    print(img_name)
        #    fname = os.path.basename(img_name)
        #    line.append(line)

if __name__ == "__main__":
    train_list = './dataset/train.txt'
    train_dir = './dataset/'
    generate(train_dir, 'imgs', 'labels')