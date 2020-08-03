# Inception系列

## Inception V1

- approximating the expected optimal sparse structure
by readily available dense building blocks is a viable method for improving neural networks for
computer vision. 
  - 为了解决overfitting和计算成本这两个问题，稀疏深层网络比全连接更有效
但是，非均匀稀疏结构的运算并不是很有效率，尽管计算量减少了100倍但是查找和cache缺失却占了主要地位。另外，均匀结构和大量filters以及batch更能够有效利用密集计算。
  - Inception结构的主要思想是基于找出卷积网络中的最优局部稀疏结构如何被现成的密集组件逼近和覆盖。
- 提出Inception 模块
  - 1X1卷积先降维，再使用3X3，减少参数量和计算量
  ![](./image/19.PNG)

- GoogLeNet 整体架构

  ![](./image/18.PNG)

  ![](./image/20.PNG)


## Inception V2/v3

### 主要内容

- Reduce representational bottleneck
  - 当卷积没有显著改变输入的尺寸时，神经网络表现得更好。减少维度太多可能会导致信息丢失，这被称为representational bottleneck.

- Using smart factorization methods
  - V1中5X5卷积换成两个3X3卷积，感受野不变，计算量减少2.78倍
    ![](./image/21.PNG)
    ![](./image/22.PNG)
  - 1XN+NX1卷积代替NXN卷积：factorize convolutions of filter size nxn to a combination of 1xn and nx1 convolutions. For example, a 3x3 convolution is equivalent to first performing a 1x3 convolution, and then performing a 3x1 convolution on its output.
    ![](./image/24.PNG)
- 扩展模块的宽度而不是深度以去除representational bottleneck.
  
    ![](./image/23.PNG)

### 基本架构

![](./image/25.PNG)
