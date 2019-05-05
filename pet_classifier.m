function yguess = pet_classifier(split,neurons,filter,epochs)

%change to local addresses of following files
catsfolder = '_______\catsfolder';
dogsfolder = '_______\dogsfolder';
petsfolder = '_______\petsfolder';

%split = 800; %number of train files/1000 each
%neurons = 16; %number of neurons
%filter = 6; %nxn filter -> (n-1)x(n-1) pooling -> (n-2)x(n-2) pooling
%epochs = 10;%number of cycles

%create image datastore object
imds1 = imageDatastore(catsfolder,'FileExtensions','.jpg');
imds1.Labels = categorical(-ones(1000,1));

imds2 = imageDatastore(dogsfolder,'FileExtensions','.jpg');
imds2.Labels = categorical(ones(1000,1));

%new folder with both cats and dogs
imds = imageDatastore(petsfolder,'FileExtensions','.jpg');
imds.Files = [imds1.Files,imds2.Files];
imds.Labels = [imds1.Labels,imds2.Labels];

%checking labels
labelCount  = countEachLabel(imds);

% perm = randperm(2000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
%     %title(imds.Labels(perm(i)));
% end

%img = readimage(imds,1); % check image size = 64x64
%size(img)

%split into testing and training data
imdsSize = size(imds.Files);
[imdsTrain,imdsTest] = splitEachLabel(imds,split,'randomize');

%convolutional neural network
layers = [
    imageInputLayer([size(readimage(imds,1)) 1])
    
    convolution2dLayer(filter,neurons,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer((filter-1),'Stride',(filter-1))
    
    convolution2dLayer(filter,neurons,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer((filter-2),'Stride',(filter-2))
    
    convolution2dLayer(filter,2*neurons,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)%number of classes
    softmaxLayer
    classificationLayer
];

%training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',epochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',60, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

yguess = classify(net,imdsTest);
yguess_train = classify(net,imdsTrain);

ytest = imdsTest.Labels;
ytrain = imdsTrain.Labels;
%accuracy function from before
accuracy_test = calculate_accuracy(ytest,yguess)
accuracy_train = calculate_accuracy(ytrain,yguess_train)
%display wrongly classified images
wrongGuess = find(yguess ~= ytest);

for i = 1:length(wrongGuess)
    subplot(ceil(length(wrongGuess)/3),3,i);
    imshow(imdsTest.Files{wrongGuess(i)});
    title(wrongGuess(i));
    %title(imdsTest.Labels(wrongGuess(i)));
end
sgtitle('Misclassified pets')    
