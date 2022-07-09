# Inspecting deep learning models

## CNNs

### First layer

- Visualizing filters

### Intermediate layers

- Visualize activations
- Maximally activating patches
- Occlusion experiments: mask part of the image before feeding to CNN, draw heatmap of probability at each mask location. Try to detect which patch(es) in an image are the most important for processing that image
- Saliency maps: compute gradient of (un-normalized) class score with respect to image pixels; take the absolute value and max over RGB channels. Helps uncover biases
- Intermediate features via (guided) backprop
- Gradient ascent: create a random image, process it with the CNN, and update the image values following the gradients for a given class. Repeat this step until plateau

- DeepDream
- 
### Last layer

- Visualizing nearest neighbors of images in the feature space
- dimensionality reduction (PCA, T-SNE): given an image, take the features at the last layer. Perform T-SNE on that feature vector to get a 2D point - p; place that image on a canvas centered at coordinates $p$
- Visualizing activations

### Resources:

- https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/