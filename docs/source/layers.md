# Layers

## Convolutional layer

Parameters:
- number of output channels
- filter shape
- stride

Commonly used parameters:
- Activation: ReLU
- Initialization: Xavier/He + Kaiming; that is, scale the weights matrix by $\frac{2}{\sqrt{input_dims}}$, where $input_dims = {filter_size}^2*num_input_channels$

## Dropout

In each forard pass, randomly set some neurons to zero with probability $q=1-p$.

The activations should be rescaled consequently by a factor of $p$.

## DropConnect

## Normalization layers

### Batch normalization

#### Training

For each batch, normalize the activations (that is, the values before the non-linearity) on a per-feature basis. That is, for each feature, compute the mean and sample variance of the mini-batch, and standardize the values.

$x^{'} = \frac{x - \hat{x}}{\sqrt{var(x) + \epsilon}}$

Include an additional linear layer - with learned parameters $\gamma$, $\beta$- at the end of batch normalization to allow the possibility to restore the values as they were before normalization.

$y = \gamma x^{'} + \beta$

#### Test

Use the running mean and variance averages as registered during the training phase to perform standardization. 

#### Batch normalization for Convolutional layers

The goal is to normalize all the values in a feature map in the same way. To achieve this, the normalization is applied to each feature map / channel (as opposed to being applied to each feature).

So, for a batch of size  $b$ and a set $s$ of feature maps of size $h \times w$, we apply the BN $s$ times, with each of them processing $b \cdot h \cdot w$ elements

#### Drawbacks

For applications where the batch size is small, the variability in the sample statistics computed for each batch has a negative effect on the overall model performace. Group normalization can solve this issue.

### Layer normalization

Normalizes all the features of all the channels for each training sample.

The number of elements included in each normalization step is $num_channels \cdot channel_height \cdot channel_width$.

There are $b$ normalization operations for each batch.

### Instance normalization

Normalizes all the features in a single channel for each training sample. 

The number of elements included in each normalization step is $channel_height \cdot channel_width$.

There are $num_channels \cdot b$ normalization operations for each batch.

### Group normalization

Normalizes all the features in a group of $G$ channels for each training sample. The number of groups (or channels per group) is an hyperparameter of the model.

## Max pooling

### Fractional max pooling