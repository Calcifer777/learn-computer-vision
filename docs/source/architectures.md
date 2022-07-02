# Architectures

## AlexNet (2012)

### Layers

- Repeat(8)
  - Conv
  - Max Pooling
  - Batch Norm
- Repeat(3)
  - FC

### Details

Distributed training

## VGG (2014)

Small filters, deeper networks

- Repeat(19)
  - Conv(size=3, stride=1)
  - Max Pooling
  - Batch Norm
- Repeat(3)
  - FC

## GoogleNet (2014)

- Deeper network, computationally efficient
- 22 layers
- Efficient 'Inception' module
- No FC layers
- Less parameters than AlexNet

### Inception module

The layers inputs are processed by a set of convolutional operations, each with different parameters (conv: filter size, number of channels, max pool: stride).

To prevent the explosion of the number of parameters and of the number of ops in each layer, 1x1 convolutional bottlenecks process the input to project it into a lower dimensional space.

## ResNet

### Layers

- 152-layer model
- add residual connections in the CONV layers
- BN after every CONV layer

### Hyperparameters

- Batch size: 256
- Initialization: Xavier/2 
- Optimizer: SGD + momentum
- Learning rate: divided by 10 when validation error plateaus
- Weigth decay
- No dropout used

### Residual connections

## Stochastic depth

## Fully convolutional networks

Architecture features:
- Can take arbitrary size as inputs
- Up sampling / transposed convilution
- skip connections

Use cases:
- pixel-wise semantic segmentation


## R-CNN

### Components

- Region of Interest (ROI) proposal
  - 2000 regions
  - selective search algorithm
  - rescaling to a fixed size
- Process each ROI proposal through a (possibly pretrained) CNN
- Targets
  1. bounding box region prediction
     - adjusts the bounding box of the input ROI to match the region of the detected object
     - a 4d vector ($x_min$ $x_max$, $y_min$, $y_max$)
  2. class score prediction: SVM loss on the object classes
- Loss: a weighted sum of the losses from the two targets

### Issues

- Proposal extraction is slow and takes a lot of disk space
- All the proposals have to be resized and passed through the network, which also adds an overhead
- Considering that the algorithm is executed on cpu, the inference time becomes slow. 

### Fast R-CNN

Change w.r.t. plain R-CNN
- the ROI are computed on a intermediate feature map.

Issues
- ROI proposal calculations are a bottleneck during prediction

### Faster R-CNN

Computing proposals with a deep convolutional neural network leads to an elegant and effective solution where proposal computation is nearly cost-free given the detection network’s computation. To this end, we introduce novel Region Proposal Networks (RPNs) that share convolutional layers with state-of-the-art object detection networks. By sharing convolutions at test-time, the marginal cost for computing proposals is small (e.g., 10ms per image).

The key observation is that the convolutional feature maps used by region-based detectors, like Fast R- CNN, can also be used for generating region proposals. An RPN can be placed on top of these convolutional features by adding a few additional convolutional layers, so that it simultaneously regress region bounds and objectness scores at each location on a regular grid.

To unify RPNs with Fast R-CNN object detection networks, we propose a training scheme that alternates between fine-tuning for the region proposal task and then fine-tuning for object detection, while keeping the proposals fixed.

#### RPNs

A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of rectangular object proposals, each with an objectness score. We model this process with a fully convolutional network. Because our ultimate goal is to share computation with a Fast R-CNN object detection network, we assume that both nets share a common set of convolutional layers. 

To generate region proposals, we slide a small network over the convolutional feature map output by the last shared convolutional layer. This small network takes as input an $n \times n$ spatial window of the input convolutional feature map. Each sliding window is mapped to a lower-dimensional feature. This feature is fed into two sibling fully-connected layers—a box-regression layer (reg) and a box-classification layer (cls).

This architecture is naturally implemented with an n × n convolutional layer followed by two sibling $1 \cross 1$ convolutional layers (for *reg* and *cls*, respectively).

#### Anchors

At each sliding-window location, we simultaneously predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as $k$.

By default we use 3 scales and 3 aspect ratios, yielding k = 9 anchors at each sliding position.

An important property of our approach is that it is *translation invariant*. This property also reduces the model size and it has less risk of overfitting on small datasets.

#### RPN Loss function

For training RPNs, we assign a binary class label (of being an object or not) to each anchor. We assign a positive label to two kinds of anchors:
- the anchor/anchors with the highest IoU overlap with a ground-truth box
- an anchor that has an IoU overlap higher than 0.7 with any ground-truth box

The loss function is a weighted sum of the regression loss and the classifier loss

#### Handling multiple ROI sizes

The features used for regression are of the same spatial size (3 × 3) on the feature maps. To account for varying sizes, a set of k bounding-box regressors are learned. Each regressor is responsible for one scale and one aspect ratio, and the k regressors do not share weights. As such, it is still possible to predict boxes of various sizes even though the features are of a fixed size/scale, thanks to the design of anchors.

#### Training

**'Image centric' sampling strategy**

Each mini-batch arises from a single image that contains many positive and negative example anchors. 

Randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1.

#### Predictions

The algorithm allows predictions that are larger than the underlying receptive field.  Such predictions are not impossible—one may still roughly infer the extent of an object if only the middle of the object is visible.

Some RPN proposals highly overlap with each other. To reduce redundancy, we adopt non-maximum suppression (NMS) on the proposal regions based on their cls scores.

#### Resources

- Code: https://github.com/rbgirshick/py-faster-rcnn

### Mask R-CNN

## Yolo


