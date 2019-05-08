%This function takes in a vector of true labels ytrue
%and a vector of guessed labels yguess and reports back
%the accuracy of the guesses out of 100%.
function accuracy = calculate_accuracy(ytrue,yguess)

n = length(ytrue);
accuracy = 100/n*sum([yguess == ytrue]);
