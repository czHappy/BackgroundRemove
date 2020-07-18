# Sequence Model

## Recurrent Neural Networks
### Sequence Models

- Speech recognition 
- Music  generation
- Sentiment classfication
- DNA sequence analysis
- Machine translation
- vidio activity recognition
- name entity recognition

    ![](./image/152.png)

### Notation
- $x^{(i)<t>}$ 
- $y^{(i)<t>}$ 
- $T_x^{(i)}$ length of $x^{(i)}$
- $T_y^{(i)}$  length of $y^{(i)}$

    ![](./image/153.png)

- Vocabulary Representation
  - 30000~50000 words
  - one-hot encoding

    ![](./image/154.png)

### RNN
- Why not a standard network?
  - I/O can be different lengths in different examples.
  - Doesn't share features learned across different positions of text.
- RNN

    ![](./image/155.png)
    ![](./image/156.png)
    ![](./image/157.png)

### Backpropagation through time
  
![](./image/158.png)