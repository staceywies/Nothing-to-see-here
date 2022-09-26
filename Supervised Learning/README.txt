# CS7641, assignment 1, Supervised Learning
Stacey Wieseneck, swieseneck3@gatech.edu

code can be opened at: https://github.com/staceywies/Nothing-to-see-here/tree/main/Supervised%20Learning

Going through my code the relevant files for recreating the data and graphs seen in my write up are as follows:
- DecisionTree.py: old code from a previous class that I used as my decision tree learner
- DecisionTree1.py: old code slightly altered just for use in the boosting portion of code
- Decision Tree.ipynb: Jupyter notebook used to engage with the DecisionTree class above, used to create graphs as well
- NeuralNetowrks.ipynb: Jupyter notebook used for all neural network coding and graph creation
- Boosting.ipynb: Jupyter notebook used for all boosting coding, also references the DecisionTree1.py as previously stated
- SVM.ipynb: Jupyter notebook used for all SVM calculations and graph creation
- KNN.ipynb: Jupyter notebook used for all KNN calculations and graph creation
- energy.csv: last but not least the data set used, sourced from https://archive.ics.uci.edu/ml/datasets/Energy+efficiency

#####Decision Trees#####
Starting with decision trees, the class created takes in training data with the input and output seperately to build a model.
It accepts input information to then return query values.
When building the decision tree the split is done very simply by looking at the correlation between output and each indvidual input factor. The factor with the highest correlation is used and split on the mean of it's value.
Pruning is also very simply done, the DecisionTree class is coded so that no redundant leaves are formed if the subset of a data on a branch have all of the same output factors. Meaning it will not only form leaves to the specified leaf size, but also larger ones if they have all of the same resulting value.

The decision tree was run at varying amounts of test/train data split. Time, iterations, error rates for test and train were recorded and graphed. This is seen in the Jupyter notebook.

#####Neural Networks#####
All neural network work is done within the jupyter notebook.
Weights start with a value of 1 and then gradient descent is iterated over a single layer neural network with all inputs (8 + 1 layer added for bias) and one output.
Initial values for weights are all set to 1.
Learnig rate is set to 0.0000002.

#####Boosting#####
Boosting is done in it's dedicated jupyter notebook. There is another .py file for the decision tree learner specific to boosting that I originally started with as well as an archived jupyter notebook if you want a laugh. 
Boosting distribution starts with even distribution across all variables.
It is iterated 50 times and scikit package is used- relevant documentation can be found here:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
A decision tree learner is used.

#####SVM#####
SVM work is done with the sklearn package, specifically svm for documentation can be found here:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
This package was a very simple set up that takes in test data, and allows for the user to vary the kernel used. 
The kernels used were linear as well as polynomial with degree of 8.
This was also iterated for a different splits between the test and training data.

#####KNN#####
K-Nearest Neighbors code was done entirely in it's dedicated jupyter notebook.
The code is almost line for line copied from:https://realpython.com/knn-python/ (Reading the assignment instructions I believe this is allowed/expected that we steal or copy this but felt wierd to do)
linalog is used as the distance function for calculated the "nearest" part of KNN.
Different values of K were iterated and error rates were calculated for each.