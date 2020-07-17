from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, add, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model

from sklearn.metrics import log_loss

from load_data import load_cifar10_data

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
        直接相加，并不需要1*1卷积
        参数：
        input_tensor: 输入
        kernel_size: 卷积核大小
        filters: 卷积核个数，需要按顺序指定3个，例如（64,64,256）
        stage和block:主要为了绘图观察，指定好名字 
    """

    nb_filter1, nb_filter2, nb_filter3 = filters # 取出3个卷积部分的卷积核数量
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    x = add([x, input_tensor], name = conv_name_base +'_add')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
        需要1*1卷积
        输入参数与identity_block一致
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Conv2D filter数量，卷积核大小，strides，name等参数
    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a')(input_tensor)
    # BN操作，按照channel那一维度，NHWC,故而bn_axis设置为3
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # relu激活函数
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # shortcut和第三部分卷积的卷积核数量相等，故而是nb_filter3,且直接取input_tensor作为输入
    # 为了匹配维度，需要用1X1卷积做projection
    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    #shortcut和卷积相加
    x = add([x, shortcut], name=conv_name_base +'_add')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_rows, img_cols, color_type=3, num_classes=None):
    """
    
        网络模型定义，需要下载好权重参数文件
        
    """
    #按照Tensorflow格式
    global bn_axis
    bn_axis = 3
    img_input = Input(shape=(img_rows, img_cols, color_type))
    
    #网络模型定义
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 全连接层
    # 平均池化
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    # flatten用于将输入的多维数据压成一维数据，用于输出到全连接层
    x_fc = Flatten()(x_fc)
    # Dense层为全连接层
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # 创建模型 Model函数式 ，传入输入和输出对象
    model = Model(img_input, x_fc)

    # 加载权重参数
    """
    for layer in model.layers:
        layer.trainable = False
        model.layers[:100]
    """
    weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path)


    # 迁移学习时全连接层用自己的开始学习
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # 创建新的模型
    model = Model(img_input, x_newfc)
    # 训练配置
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True) # SGD优化器
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  
    return model

if __name__ == '__main__':

    # 定义好输入的数据
    img_rows, img_cols = 224, 224  # h,w
    channel = 3
    num_classes = 10 
    batch_size = 16 
    nb_epoch = 10

    # 加载数据
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # 网络模型 img_rows, img_cols, color_type=3, num_classes=None
    model = resnet50_model(img_rows, img_cols, channel, num_classes)
    # Converts a Keras model to dot format and save to a file.
    #plot_model(model, to_file='model_new.png', show_shapes=True)

    # Fine-tuning
    H = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1, #日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
              validation_data=(X_valid, Y_valid),
              )

    
    
plt.style.use("ggplot")
plt.figure()
plt.ylim(0, 3)
N = nb_epoch
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('result_new.png')