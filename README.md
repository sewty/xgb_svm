# xgb_svm
Hyper-parameter tuning for XGB and SVM models

This was my submission to an assignment for my Machine Learning course. I received a list of learning models and was required to choose two, create and fit them to the given dataset (a9a.txt and a9a.t) and then slowly tune the hyper-parameters of each model to improve their accuracy.

The first half of the script establishes the train/test split, final dataset, and functions to train the models, test the models, and tune each relevant hyper-parameter. The tuning functions essentially take in a range of values and test each of them with cross-validation to find the best value within the given range. As you get better accuracy, you run the function again with a range centered around the previous best result. This process could certainly be iterated with something like a for loop, but I chose to do it manually for the sake of runtime.

The second half of the script is the actual driver code I used to get my results. You can just uncomment them and run the file and you should receive very similar, if not the same, results. I wrote multiple warnings within the script that the C tuning for SVMs can take a while, so be careful of the numbers you pass in (if you decide to not use the default ones).
