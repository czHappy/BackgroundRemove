# Mechine Learning
<!-- TOC -->

- [Mechine Learning](#mechine-learning)
  - [Tools and Knowledge](#tools-and-knowledge)
    - [Python](#python)
    - [Project](#project)
  - [Mechine Learning](#mechine-learning-1)
    - [Liner regression with one variable](#liner-regression-with-one-variable)
    - [Liner regression with Multiple variables](#liner-regression-with-multiple-variables)
    - [Logistic Regression](#logistic-regression)
    - [Regularization](#regularization)
  - [What is Neural Networks](#what-is-neural-networks)
    - [Housing Price Prediction](#housing-price-prediction)
    - [Neural Network](#neural-network)
    - [Backward Propagation](#backward-propagation)
    - [Support Vector Mechine (SVM)](#support-vector-mechine-svm)
  - [Principle Analyses](#principle-analyses)

<!-- /TOC -->
## Tools and Knowledge
### Python
- tensorflow basic operation 
  - Chap19, deep learning and Tensorflow2 Practice 
  - [Course](https://study.163.com/course/courseMain.htm?share=1&shareId=1023679146&courseId=1003606092&_trace_c_p_k2_=79cb3bb61bd14afd9a1accbeeeffb856)
- tensorflow advanced operation 
  - [Course](https://study.163.com/course/courseMain.htm?courseId=1005214052&share=1&shareId=1023679146)
- Numpy
  - Python and data science practice
  - [Course](https://study.163.com/course/courseMain.htm?courseId=1004786005&share=1&shareId=1023679146)


### Project
- style-transfer
  - Chap4, Deep learning-Tensorflow-Practice
  - [Course](https://study.163.com/course/courseMain.htm?courseId=1005214052&share=1&shareId=1023679146)

  
## Mechine Learning 
### Liner regression with one variable
- Model representation(House Price predict)
- Supervise Learning:Know right answer for each example 
- Regression: Predict output.
- Training set: Use it to learn features so that mechine knows how to get output.
  -  A pair (x(i),y(i)) is called a training example, and the dataset that we’ll be using to learn—a list of m training examples (x(i),y(i));i=1,...,m—is called a training set. 
- Hypothesis Funciton:H(x) h
  - Liner: $h_{\theta}$ = $\theta_0$ + $\theta_1$x
  - How to choose proper parameters ?
  - Target Function/ Cost Funcitons
    - J($\theta_0$ , $\theta_1$) = Min Squre error

    ![](./image/1.png)
- Gradient descent
    - convergence
    - learning rate: $\alpha$
      - if α is too small , descent can be slow; while α is too large,overshoot the minimum or even diverge.
    - Local minimum
      - Derivative equals to 0. Do nothing.
      - When approach a local munium, GD will automatically take smaller steps, no need to decrease α over time.
      - Bowl-shaped:Global optimum.
    - 
    ![](./image/2.png)


### Liner regression with Multiple variables
- Data set Presentation


    ![](./image/3.png)
    ![](./image/4.png)

- Gradient descent of Mutiple variables.
  - Basic Algorithm

    ![](./image/5.png)
    ![](./image/6.png)
  - Feature Scaling
    - Make sure features are on a similar scale so that make convergence quicker.
    - $$-1 \leq x_i \leq 1 $$
    - $$X_i = {{X_i - \mu_i} \over S_i} , \mu_i = \sum_{set}, S_i = Set(Max-Min)$$
- Convergence
  - Declare convergence if J decreases by less than $\epsilon$ in one iteration.

- Polynomial regression
  - Note:It maybe use Feature Scaling.

    ![](./image/7.png)
- Normal equation
  - $$\theta = (X^TX)^{-1}X^Ty  )$$
  - Matrix Reversible
    - O(n^3) slow if n is very large.
  - Irreversiable(less common)
    - Delete some features or use regularization.
    - Delete some Reduntant features(linearly dependent) to ensure the matrix is reversible.
    ![](./image/8.png)

### Logistic Regression
- Classfication.
  - $$h_{\theta}(x) = {1 \over {1+e^{-\theta^Tx}}},h_{\theta}(x) = P(y=1|x;\theta)$$
  - Fit the parameter to the data set.
  - Suppose predict "y=1" if $h_{\theta}(x) \geq 0.5$,else y=0. Just $\theta^Tx \geq 0$. So the question has been transformed to Linear regression. 

    ![](./image/10.png)
    ![](./image/11.png)
- New Cost Function
  - Use Linear regression cost function may cause many local optimum, because it not a convex function.
  - Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression. 

    ![](./image/14.png)
    ![](./image/12.png)

- Advanced optimization algorithm
  - Given $\theta$, how to compute  $J(\theta)  and  {\partial \over \partial\theta_i}J(\theta)$
  - GD
  - Conjugate gradient
  - BFGS
  - L-BFGS
  - No need to manually pick $\alpha$ and often faster than GD, but more complex.
  
- Multiple Classfication 
  - (one-versus-all/one-versus-restr)
  
  ![](./image/15.png)

### Regularization
- overrfitting
  - Phenomenon
    ![](./image/16.png)

  - Solve Options
    - Reduce features
    - Regularization
      - Keep all the features
      - Works well when we have a lot of featuers, each of which contributes a bit to predicting y.
  
    ![](./image/17.png)

- Regularized Linear Regression


    ![](./image/18.png)
    ![](./image/19.png)
    
- Regularized Logistic Regression


    ![](./image/20.png)
    ![](./image/21.png)

    ---
## What is Neural Networks
### Housing Price Prediction

- Bias unit: $x_0$=1
- Input wires and Output wires
- Activation function:non-linear function g(z)
- Weight:Parameters of model.

  ![](./image/22.png)

### Neural Network
- A Neural Network is just a group of some different Neurons strung together.
  
  ![](./image/23.png)

- Forward propagation
  - Layer data passes by Metrix Parametered Sigmod function and do iteration jobs.
  
  ![](./image/24.png)

  - Vectorized implementation.

  ![](./image/24.png)
  ![](./image/25.png)

- Simple Example
  - XOR AND NOT XNOR opertation
    ![](./image/27.png)

- Application: Multiple Classfication
  
  ![](./image/26.png)


### Backward Propagation
- Fitting the parameters of the Neural Network

  ![](./image/28.png)
-  Cost Function
  
  ![](./image/29.png)

- Signs Summary
  - $a^{(j)}$,node vector of layer $j^{th}$
    - $a_i^{(j)}$ ,$i^{th}$ element of $a^{(j)}$ 
    - $a^{(j)} = g(z^{(j)})$, $z^{(j)}$ is middle value which needs  parameters matrix and node vector of layer j-1.
  - $\Theta^j$, matrix of parameters controlling function mapping from layer j to layer j+1, which helps get $z^{j+1}$ by calculating dot product of $\Theta^{j}a^{j}$.

  - $z^{i}$, middle value vector, parameters of sigmid function g(z), which calculates dot product of $\Theta^{i-1}a^{i-1}$

  - $h_\Theta(x)$ is value of the last sigmod function parametered by $z^{(last\_layer)}$,or the output vector of the neural network.
  - $\delta^{(l)}_j$ = error of node j in layer l. Or $\Delta$
    - $\delta^{(4)}_j = a^{(4)}_j - y^i$ 
  
- Gradient Computation
  - Given one training example(x,y),forward propagation:
  
  ![](./image/30.png)
- BP algorithm
  - Algorithm
  
    ![](./image/31.png)
  - About $\delta$
  
    ![](./image/32.png)

- Random Initialization
  -  Zero initialization:After each update,  parameters corresponding to inputs going into each of two hidden units are identical.
  -  Initial each $\Theta^(l)_{ij}$ to a random value in [$-\epsilon,\epsilon$]



### Support Vector Mechine (SVM)
- Optimize target
  ![](./image/161.png)
  ![](./image/162.png)
  ![](./image/163.png)

- SVM: Large margin classfier
  - More robust to odd points when parameter C is small.
   
    ![](./image/164.png)
    ![](./image/165.png)

  - Kernel function: Goto high feature space so that these points are linear separable. 
  
    ![](./image/165.png)
##  Principle Analyses 
- Liner refression:Why use partial derivatives to decide where to go?
  - [First order Taylor formula expansion]( https://zhuanlan.zhihu.com/p/86147140)


- Why Logistic regression's formular of gradient descent is similar to linear regression's?
  - [maximum likelihood estimation MLE](https://blog.csdn.net/zjuPeco/article/details/77165974)
- Why using Regularization can prevent overfitting?
  - [James-Stein estimator, make a tradeoff between bias and variance.](https://www.zhihu.com/question/20700829)

- The math principle behind BP Algorithm
  - [Matrix calculus and Vector calculus](https://zhuanlan.zhihu.com/p/22473137)




