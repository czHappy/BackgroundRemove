# Convolution Neural Networks


## Introduce
### Computer vision problems
- Application
  - Image Classfication
  - Object detection
  - Neural Style Transfer
  - ...

- Deep learning on Large images
  
  ![](./image/90.png)

### Edge detection 
- examples

    ![](./image/92.png)

- How to do convolution ooperation ?
  - Cross-correlation instead of convolution, but in typical convolution literature it is.
    - Skip the mirroring operation.

  ![](./image/91.png)
  ![](./image/93.png)

- Filter
  - Result Size: (n-f+1)X(n-f+1),usually f is odd.


    ![](./image/94.png)

  - Downsides
    - Shink: Output matrix will smaller and smaller.
    - Pixels in the corner lack of overlaps. You're are throwing away a lot of information near the edge of the image.
  
- Padding
  - Pad the image with an aditional border of one pixel all around the edges.
  - Pad with 0.
  
    ![](./image/95.png)

- Stride
  - Move distence once a time.
  - Result size = $\lfloor {n-f+2p \over s}+ 1 \rfloor$

    ![](./image/97.png)

### Convolutions
- Convolutions on RGB images: Convolve on cubes

    ![](./image/98.png)

- Multiple fiters
  
    ![](./image/99.png)

- Single layer of a convolution network
  - Notation
  
    ![](./image/102.png)

  - parameters: Filter matrix and bias
  
    ![](./image/100.png)

  - No matter how many features the input has, the number of parameters remains fixed corresponding to the size and quantity of filters.

    ![](./image/101.png)

### ConvNet 
- example

    ![](./image/103.png)
- Types of layers in Convolution network
  - Convolution CONN
  - Pooling POOL
  - Fully connect FC

- Pooling layers
  - Max pooing(more ofen)
  - Average pooling
  
    ![](./image/104.png)

- LeNet-5

    ![](./image/105.png)
    ![](./image/106.png)

### Why convolutions? 
- Less Parameters

    ![](./image/107.png)

- Translation invariance
- Deep learning and Convolution networks


    ![](./image/108.png)


## Classic neural work
### LeNet-5

![](./image/109.png)

### AlexNet

![](./image/110.png)

### VGG-16
- It uses a much simpler network where you focus on just having conv layers.

- VGG-16 does almost as well as VGG-19

### Residual Network

- Residual block


  ![](./image/112.png)

- ResNet has certainly been effective at helping train very deep networks.

  ![](./image/113.png)

- Principal 
  - For neural network, identity maps are not easy to fit. 

  ![](./image/114.png)


### Network in network
- 1X1 convolution
  - Reduce Dimensions
  - Full Connect


  ![](./image/115.jpg)

- Scalling( Reduce parameters through bottleneck )
  
  ![](./image/117.PNG)
  ![](./image/118.PNG)


- Inception Network
  
  ![](./image/116.PNG)

### Advice
- Github open-source implemention
- Transfer learning, get weight parameters that have been trained.
  - Use them as initialization 


- Data augmentation
  - Mirroring on the vertical axis.
  - Random Cropping
  - Color shifting :More robust to changes in the colors of your images.

  ![](./image/121.PNG)

## Objection Detection
### Introduce
- Classfication with localization
 
  ![](./image/122.PNG)


- Defining the target label y
  - Pc  logistic cost function
  - Position  square error function

  ![](./image/123.PNG)

- Landmark detection
  
  ![](./image/124.PNG)

### Sliding windows algorithm
- Object Detection
  - Training set:Cropped Images
  - Train CovNet
  - Slide the window with a fixed stride,  Run the CovNet to predict if it is a car.(Too slow)

  ![](./image/125.PNG)

- Sliding windows algorithm in CovNet
  - Turning FC layer into convolution layers
  - In fact, Convolution concludes sliding windows. Use this feature to avoid repetitive computations.

  ![](./image/126.PNG)
  ![](./image/127.PNG)
  ![](./image/128.PNG)

### YOLO Algorithm
- Bounding Box

  ![](./image/129.PNG)

- Intersection over Union
  
  ![](./image/130.PNG)

- Non-max supression
  
  ![](./image/131.PNG)

- Anchor Boxes
  - Make algorothm more specialized.
  ![](./image/132.PNG)

  - In fact, the chance of two objects have the same midpoint is very low.

- YOLO algorithm
  
  ![](./image/133.PNG)
  ![](./image/134.PNG)

- R-CNN
  - Region Proposal
  - Segmentation algorithm


## Application
### one-shot
- Recognize a person given just one single image.
- Similarty function: d(image1, image2)  $\leq \tau$
  - Travesal and compare

### Siamese network
- Concept

  ![](./image/137.PNG)

- How to traning the special NN?

  ![](./image/138.PNG)

### Triplet Function
- Definition of d
  ![](./image/139.PNG)

- Training
  
  ![](./image/140.PNG)
  
- Choose triplets that are hard to train on.

  ![](./image/141.PNG)
  ![](./image/142.PNG)

- Work
  
  ![](./image/143.PNG)

### Style Transfer
- Introduce

  ![](./image/144.PNG)

- Deep convolution Networks


- Cost function
  - $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$

    ![](./image/145.PNG)
    ![](./image/146.PNG)

- Content cost function

  ![](./image/147.PNG)


- Style cost function
  
  ![](./image/148.PNG)
  ![](./image/149.PNG)

### 1D 2D 3D
 
  ![](./image/151.PNG)
  ![](./image/150.PNG)

## Questions
### Why Residual network works?
- [Identity maps](https://zhuanlan.zhihu.com/p/80226180)

### YOLO
- [YOLO](https://zhuanlan.zhihu.com/p/32525231)