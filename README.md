# MATLAB-Pet-Classifier
EK381 A1 Goyal
Machine Learning Challenge (Spring 2019) - 1st Place

CNN (convolutional neural network) that attempts to classifies cats and dogs using provided dataset (cats = -1, dogs = +1)

1. Unzip
2. Open pet_classifier.m and replace locations of 
	catsfolder, dogsfolder and petsfolder with local locations
3. Add unzipped folder to current MATLAB path so calculate_accuracy.m is there
4. Run pet_classifier.m

ex. >>yguess = pet_classifier(800,16,6,10);

yguess = pet_classifier(split,neurons,filter,epochs)
split - number of training images for a 1000 cat + 1000 dog dataset
neurons - number of neurons in initial layer
filter - pooling layers are nxn to (n-1)x(n-1) to (n-2)x(n-2)
epochs - number of cycles with reshuffling of images

optimized using manual permutation (nested loop through values of input parameters split,neurons,filter) and plotted guess accuracy w.r.t. each pair

future version will use gradient descent for more thorough optimization
