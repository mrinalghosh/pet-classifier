# pet-classifier
EK381 Machine Learning Challenge (Spring 2019) - 1st Place

Convolutional neural network that classifies cats and dogs using provided dataset

1. Unzip
2. Open pet_classifier.m and replace locations of 
	catsfolder, dogsfolder and petsfolder with local locations
3. Add unzipped folder to current MATLAB path to include calculate_accuracy.m
4. Run pet_classifier.m

```matlab
yguess = pet_classifier(split,neurons,filter,epochs)
```

Parameters:
* split - number of training images for a 1000 cat + 1000 dog dataset
* neurons - number of neurons in initial layer
* filter - pooling layers are n x n to (n-1) x (n-1) to (n-2) x (n-2)
* epochs - number of cycles with reshuffling of images
