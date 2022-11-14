# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: perceptron.py
# SPECIFICATION: This program is designed to simulate a grid search of some hyperparameters of single-layer and
#                   multi-layer perceptron. The performance of these combination of hyperparameters is base on their
#                   accuracy. The accuracy will be updated and printed for each interation if the combination of
#                   hyperparameters leads to greater accuracy than what was calculated from the previous interation.
#                   The two hyperparameters that will be used is the learning rate and shuffle.
#
#                   NOTE: Used PyCharm 2022.2.1 to write code for this project. Current Python version installed
#                           is 3.10.1, and current scikit-learn version is 1.1.2. My installed version of sklearn does
#                           not have the "n_iter" parameter for the Perceptron model. From looking at the version
#                           suggested on the comment portion of line 22, "n_iter" is for the number of passes of the
#                           training data. For scikit-learn version 1.1.2, "max_iter" is what is used. The default value
#                           for "max_iter" is 1000, so I have chosen to not include that parameter for line 53, since
#                           it is the same number of iterations as what was written for "n_iter"
# FOR: CS 4210.01 - Assignment #4
# TIME SPENT: It took ~1 hour to write the code but during testing I discovered the issue with the "n_iter" parameter
#               and researched how to correct that issue, as described in my NOTE section above.
#               I spent around two hours total to finish this project.
# -----------------------------------------------------------*/
import sklearn
# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  # pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # list for the learning rates hyperparameters
r = [True, False]  # list for the Shuffle hyperparameter

df = pd.read_csv('optdigits.tra', sep=',', header=None)  # reading the data by using Pandas library

X_training = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:, -1]   # getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None)  # reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]  # getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]  # getting the last field to form the class label for test

num_of_correct_predictions = 0  # initialize variable to hold the number of correct predictions for each model
perceptron_accuracy = 0.0  # initialize variable to hold the highest accuracy found for Perceptron
MLPClassifier_accuracy = 0.0  # initialize variable to hold the highest accuracy found for MLPClassifier
current_accuracy = 0.0  # initialize variable to hold the calculated accuracy of the current interation.

length_X_test = len(X_test)  # Use the length of the test set to calculate the accuracy of each model.
for w in n:  # iterates over n
    for b in r:  # iterates over r
        for a in range(2):  # iterates over the algorithms
            # Create a Neural Network classifier
            if a == 0:
                clf = Perceptron(eta0=w, shuffle=b)  # eta0 = learning rate, shuffle = shuffle the training data
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(25,),
                                    shuffle=b, max_iter=1000)
                                    # learning_rate_init = learning rate,
                                    # hidden_layer_sizes = number of neurons in the ith hidden layer,
                                    # shuffle = shuffle the training data
            clf.fit(X_training, y_training)  # Fit the Neural Network to the training data

            # make the classifier prediction for each test sample and start computing its accuracy
            # hint: to iterate over two collections simultaneously with zip() Example:
            # for (x_testSample, y_testSample) in zip(X_test, y_test):
            # to make a prediction do: clf.predict([x_testSample])
            # --> add your Python code here
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                predicted = clf.predict([x_testSample])
                if predicted == y_testSample:
                    if a == 0:  # tally correct predictions for Perceptron
                        num_of_correct_predictions += 1
                    else:  # tally correct predictions for MLPClassifier
                        num_of_correct_predictions += 1

            # check if the calculated accuracy is higher than the previously one calculated for each classifier.
            # If so, update the highest accuracy and print it together with the network hyperparameters.
            # Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            # Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            # --> add your Python code here
            if a == 0:  # calculate the current accuracy for Perceptron
                current_accuracy = num_of_correct_predictions/length_X_test
            else:  # calculate the current accuracy for MLPClassifier
                current_accuracy = num_of_correct_predictions/length_X_test

            if w == 0:  # if on the first interation of the outer loop
                if a == 0:  # assign current accuracy for Perceptron and print relevant info
                    perceptron_accuracy = current_accuracy
                    print("\nHighest accuracy for Perception so far:  " + str(perceptron_accuracy))
                    print("Hyperparameters: Learning Rate = " + str(w) + ", Shuffle = " + str(b))
                    print("==========================================================================\n")
                else:  # assign current accuracy for MLPClassifier and print relevant info
                    MLPClassifier_accuracy = current_accuracy
                    print("Highest accuracy for MLPClassifier so far:  " + str(perceptron_accuracy))
                    print("Hyperparameters: Learning Rate = " + str(w) + ", Shuffle = " + str(b))
                    print("==========================================================================\n")
            else:  # if NOT on the first interation of the outermost loop
                if a == 0 and perceptron_accuracy < current_accuracy:  # check if current accuracy for perceptron is
                    perceptron_accuracy = current_accuracy             # greater than previous, if true update and print
                    print("Highest accuracy for Perception so far:  " + str(perceptron_accuracy))
                    print("Hyperparameters: Learning Rate = " + str(w) + ", Shuffle = " + str(b))
                    print("==========================================================================\n")
                elif a != 0 and MLPClassifier_accuracy < current_accuracy:  # check if current accuracy for
                    MLPClassifier_accuracy = current_accuracy  # MLPClassifier is greater than previous, update/print
                    print("Highest accuracy for MLPClassifier so far:  " + str(perceptron_accuracy))
                    print("Hyperparameters: Learning Rate = " + str(w) + ", Shuffle = " + str(b))
                    print("==========================================================================\n")

            # reset following variables to zero for reuse
            num_of_correct_predictions = 0
            current_accuracy = 0.0

        # END OF INNER LOOP
    # END OF MIDDLE LOOP
# END OF OUTER LOOP
print("END OF ALL LOOPS. LAST STATED ACCURACIES IS THE HIGHEST FOUND WITH THE GIVEN HYPERPARAMETERS.")
