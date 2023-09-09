# FID calculation details

The FID calculation involves many steps that can produce inconsistencies in the final metric. 

## What is FID?

The Fréchet inception distance (FID) is a metric used to assess the quality of images created by a generative model, like a generative adversarial network (GAN). 
Unlike the earlier inception score (IS), which evaluates only the distribution of generated images, the FID compares the distribution of generated images with the distribution of a set of real images ("ground truth").

### Calculation steps

1) Load and preprocess images from two datasets
2) Get images features from InceptionV3 model
3) Calculate mean and variances for every feature set of features
4) Calculate the Fréchet inception distance between two probability distributions

TO-DO