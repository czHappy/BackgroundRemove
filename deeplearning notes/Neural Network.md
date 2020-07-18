# Neural Networks and Deep learning
<!-- TOC -->

- [Neural Networks and Deep learning](#neural-networks-and-deep-learning)
  - [Concept](#concept)
    - [Supervised Learning](#supervised-learning)
    - [Non-Supervised learning](#non-supervised-learning)
    - [Standard NN](#standard-nn)
    - [CNN](#cnn)
    - [RNN](#rnn)
    - [Data](#data)
  - [Binary classification---Logistic Regression](#binary-classification---logistic-regression)
    - [Symbolic Representation](#symbolic-representation)
    - [How to use Gradient Descent](#how-to-use-gradient-descent)
    - [Vectorized](#vectorized)
    - [The explanation of logistic regression cost function](#the-explanation-of-logistic-regression-cost-function)
  - [Neural Networks](#neural-networks)
    - [Representation](#representation)
    - [Vectorized](#vectorized-1)
    - [Active Function](#active-function)
    - [Forward and backward propagation](#forward-and-backward-propagation)
    - [Why deep representations?](#why-deep-representations)
    - [parameters and hyperparameters](#parameters-and-hyperparameters)
    - [Deep learning and Brain.](#deep-learning-and-brain)
  - [Setting up Neural networks](#setting-up-neural-networks)
    - [Data](#data-1)
    - [Bias and Variance](#bias-and-variance)
    - [Regularization](#regularization)
    - [Normalizing inputs](#normalizing-inputs)
    - [Vanishing/exploding gradients](#vanishingexploding-gradients)
  - [Optimum Proposal](#optimum-proposal)
    - [Mini-batch gradient descent](#mini-batch-gradient-descent)
    - [Exponentially weighted averages](#exponentially-weighted-averages)
    - [Local optima](#local-optima)
  - [Tuning Process](#tuning-process)
    - [Common choice](#common-choice)
    - [Appropriate scale for hyperparameters](#appropriate-scale-for-hyperparameters)
    - [Training way](#training-way)
    - [Batch Norm](#batch-norm)
    - [Softmax](#softmax)
  - [ML Strategy](#ml-strategy)
    - [Orthogonalization](#orthogonalization)
    - [Data set division](#data-set-division)
    - [Performance of mechine and people](#performance-of-mechine-and-people)
    - [Training and testing on different distributions](#training-and-testing-on-different-distributions)
    - [End-to-End learning](#end-to-end-learning)
  - [Blog Summary](#blog-summary)

<!-- /TOC -->
## Concept
### Supervised Learning


### Non-Supervised learning


### Standard NN

### CNN
- definition
- application


### RNN

### Data
- Structure data
  - Like database
- Unstructure data
  - Audio Image Text

## Binary classification---Logistic Regression
### Symbolic Representation
- $X \in R^{n,m}$, n is the number of features and m is the number of samples.
  
- $Y \in R^{1,m}$ 
   
    ![](./image/33.png)

- Why use different cost function in Logistic regression?
  - Local minimum and Global minimum.
  
  ![](./image/34.png)

### How to use Gradient Descent
- Calculate Some important derivatives.

  ![](./image/35.png)

- Next operation for m training examples

  ![](./image/36.png)


### Vectorized
- Use vectorized calculations in numpy can take much better advantage of paralism .
- Avoid explicit for-loops.
  
  ![](./image/37.png)
  ![](./image/38.png)
  ![](./image/40.png)

- Boardcasting in python

  ![](./image/41.png)

### The explanation of logistic regression cost function

  ![](./image/42.png)
  ![](./image/43.png)



## Neural Networks
### Representation
  ![](./image/44.png)
  ![](./image/45.png)

### Vectorized

![](./image/46.png)

### Active Function
- g(z) sigmoid function
  - a = $1 \over {1+e^{-z}}$
- tanh(z) -- pretty much superior.
  - a = ${e^z-e^{-z}} \over {e^z+e^{-z}}$
- ReLU
  - g(z) tanh(z) can slow down gradient descent when the slope becomes very small.
  -  rectified linear unit
  -  a = max(0,z)
  - leaky ReLU: a = max(0.001*z , z)
- why use Non-linear function ?
  - It turns out that if you use a linear active funtion or alternatively if you don't have an active function, then no matter how many layers your neural network has, always doing is just computing a linear activation function.
  - So you might as well not have any hidden layers.

### Forward and backward propagation
- Forward propogation for layer L
  - $z^{[l]} = w^{[l]} a^{[l-1]}+b^{[l]}$
  - $a^{[l]} = g^{[l]}{(z^{[l]})}$
- Backward propogation for layer L
  
  ![](./image/48.png)
- Summary
  
  ![](./image/47.png)


### Why deep representations?
- Circuit theory and deep learning 
  - There are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute.

### parameters and hyperparameters
- Hyperparameters actually control the final values of parameters.
- It is quite possible there the best value for the learning rate a number of hidden units and so on might change(some factors such as GPUs/CPUs). Just try more times.

### Deep learning and Brain.

  ![](./image/50.png)




## Setting up Neural networks 
### Data
- training set: Training parameters

- hold-out cross validation set/development set
  - Adjust parameters, choose features and make decisions for learning algorithm. 
  - Dev set is for target.
- test 
  - Accessment
  - Make sure dev and test come from the same distribution.
  - not having a test set might be okay.

### Bias and Variance
- Train set error
- Dev set error

  ![](./image/52.png)

- Performance
  
  ![](./image/51.png)

- tradeoff

  ![](./image/53.png)

- Contrast
  - Variance small, bias small > Variance small, bias big > Variance big, bias small > Variance big, bias big
  
  ![](./image/89.png)

### Regularization
- hyperparameter $\lambda$

  ![](./image/54.png)
  ![](./image/55.png)


- Principal
  -  Non-linear activation function will be relatively linear.
  
  ![](./image/56.png)

- Dropout regularization
  - Set some probability of eliminating a node in neural network
  - Inverted dropout
  
  ![](./image/57.png)
  ![](./image/58.png)

- Other regularization
  - Data Augmentation
    - More homologous data sample.
  - Early stopping
    - Stop training once mistakes are found.
  ![](./image/59.png)


### Normalizing inputs
- Speed up training
- Two steps
  
  ![](./image/60.png)
  ![](./image/61.png)

### Vanishing/exploding gradients
- $W^{[1]}$

  ![](./image/62.png)

- Weight initialization: Make variance of weight of matrix be 1.
  
  ![](./image/63.png)

- Numerical approximation of gradients
  
  ![](./image/64.png)

- Gradients Checking

  ![](./image/65.png)


## Optimum Proposal
### Mini-batch gradient descent
- Use part of samples to update parameters in a iteration.
- Steps


  ![](./image/66.png)
  ![](./image/67.png)

- Principal
  
  ![](./image/68.png)
  
### Exponentially weighted averages
- $V_t = \beta V_{t-1} + (1-\beta)\theta_t$
  
  ![](./image/69.png)

- Bias Correction

  ![](./image/70.png)

- Gradient descent with Momentum
  - Use exponentially weighted averages to make gradient descent more stable and more quick.
  
  ![](./image/71.png)

- RMSprop---Root mean square prop

  ![](./image/72.png)

- Adam optimization algorithm
  - Adaptive moment estimation. Combine RMSprop with Momentum
  
  ![](./image/73.png)
  ![](./image/74.png)
  
- Learning rate decay

  ![](./image/76.png)
  ![](./image/75.png)


### Local optima
- In a high dimensional space, the chance of having many local optima is maybe very small, because all directions need to be same.
  
  ![](./image/77.png)
  ![](./image/78.png)


## Tuning Process
### Common choice
- Parameters
  
  ![](./image/79.png)

- Don't use a grid
- Coarse to fine
  - Zoom in to a smaller region of the hyperparameters and then sample more density within this space.

### Appropriate scale for hyperparameters
-  To sample on the log scale.
  
  ![](./image/80.png)
 
### Training way
- Pandas VS. Caviar
  
  ![](./image/81.png)

### Batch Norm
- Given some intermediate value in NN. $z^{[(l)}$
  - $\mu = {1 \over m}\sum_iz^{(i)}$
  - $\sigma^2 = {1 \over m}  \sum_i (z_i-\mu)^2$
  - $z_{norm}^{(i)}={z^{(i)}-\mu \over \sqrt{\sigma^2 + \epsilon}}$
  - $\tilde{z}^{(i)} = \gamma z_{norm}^{(i)} + \beta$
  - Parameters $\beta$ and $\gamma$ control the mean and variance of the $z^{(i)}$

- Working in NN
  
  ![](./image/82.png)

- Working with mini-batches
  
  ![](./image/83.png)
  ![](./image/84.png)

- Principal
  - Covriate shift:If the distribution of the input x changes, you might need to retrain your learning algorithm.
  
  -  What batch norm does is it reduces the amount that the distribution of these hidden unit values shifts around. It weakens the coupling between what the early layers parameters have to do and what the later layers parameters have to do, which allows the current layer  to learn by itself, a little bit more independently of other layers.

### Softmax 
- Softmax regression is a generalization of logistic regression to more than two classes.
  ![](./image/85.png)

## ML Strategy
### Orthogonalization
- The controllers do not interact with each other. In the other words, any two dimensions don't affect the same feature at the same time.

- Precision
  - In all images that classifier predicts as cats, What percentage actually are cats?
- Recall
  - Of all the images that really are cats, what percentage of actual cats are correctly recognized by your classfier?
- F1 Score
  - $2\over{{1\over p} + {1\over R}}$

- One optimizing metric and Some satisficing metrics.

### Data set division
- Keep dev and test sets come from the same distribution.
  
- Choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on.

- Size of test set
  - Set your test set to be big enough to give high confidence in the overall performance of your system.

### Performance of mechine and people
- Avoidable bias
  - Human-level error as a proxy for Bayes error.
  - Difference between Bayes error and training error.
  - Focus on bias or variance
  
  ![](./image/86.png)

- Surpass human-level
  
  ![](./image/87.png)

- Reduce bias
  
  ![](./image/88.png)


### Training and testing on different distributions
- Put data from different distributions together and Randomly shuffle them into a train,dev and test set.
  - Advantages: training dev test data set from the same distribution.
  - Disadvantage: This is setting up the dev set to tell your team to optimize for a different distribution of data than what you actually care about.
- Let training,dev and test data come from different distributions, data for dev and test comes from where your app will apply and data for training come from other place.
  - It will bring better long term performance.

### End-to-End learning 
- Pipline learning 
- Skip some middle steps and construct learning from input to output directly.
- Key: Do you have sufficient data to learn a function of the complexity needed to map x to y?
- Works quite well for machine translation.
  ![](./image/136.png)
## Blog Summary
- [part2](https://zhuanlan.zhihu.com/p/41186951)