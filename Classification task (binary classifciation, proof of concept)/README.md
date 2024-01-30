# Classification on [SAheart dataset](https://hastie.su.domains/ElemStatLearn/datasets/) using fastlane
Binary classification dataset on variable chd from: http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data
info on dataset is from: https://hastie.su.domains/ElemStatLearn/datasets/SAheart.info.txt

Data read directly from link using python and split into training and test sets in 2 separate .csv files (SAheat.csv and test.csv with the variable famhist (family history: binary variable) removed. Training and test split of 75:25.

In c++ a fastlane ann composed of 21 small relu neural networks(each with 16 units) is fitted to the data, with the output being a vector with 2 values using the softmax function to get get a probability distribution (cross-entrophy loss is used for this last layer), in total <2500 parameters. In training the loss did not go to zero entirely, so surprisingly no severe overfitting/overparameterization.

results for test set for the model: 

The average cross entrophy loss is 1.87972

The accuracy is 0.663793

The sensitivity is 0.55102

The specificity is 0.746269

