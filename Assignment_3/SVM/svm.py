# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: svm.py
# SPECIFICATION: This program is designed to simulate a grid search to find the highest accuracy of an SVM model
#                   using different hyperparameters. These parameters will be c = { 1, 5, 10, 100 },
#                   degree = { 1, 2, 3 }, kernel = { "linear", "poly", "rbf" } and
#                   decision function shape = { "ovo", "ovr" }. The program will display the current highest accuracy of
#                   the SVM model and its given combination of hyperparameters. The last displayed accuracy and
#                   hyperparameters is the highest found.
# FOR: CS 4210- Assignment #3
# TIME SPENT: ~2 hours building and testing the logic, and about 10min formatting the output
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

# defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
########################################################################################################################
df = pd.read_csv('optdigits.tra', sep=',', header=None)  # reading the training data by using Pandas library
X_training = np.array(df.values)[:, :64]  # getting the first 64 fields to create the feature training data and
                                         # convert them to NumPy array
y_training = np.array(df.values)[:, -1]  # getting the last field to create the class training data and
                                        # convert them to NumPy array
########################################################################################################################
df = pd.read_csv('optdigits.tes', sep=',', header=None)  # reading the training data by using Pandas library
X_test = np.array(df.values)[:, :64]  # getting the first 64 fields to create the feature testing data and
                                      # convert them to NumPy array
y_test = np.array(df.values)[:, -1]  # getting the last field to create the class testing data and
                                     # convert them to NumPy array
########################################################################################################################
# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
# --> add your Python code here
highest_accuracy = 0.0
for i in range(len(c)):  # iterates over c
    for j in range(len(degree)):  # iterates over degree
        for k in range(len(kernel)):  # iterates kernel
            for m in range(len(decision_function_shape)):  # iterates over decision_function_shape
                # Create an SVM classifier that will test all combinations of c, degree, kernel, and
                # decision_function_shape.
                # For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                # --> add your Python code here
                clf = svm.SVC(C=c[i], degree=degree[j], kernel=kernel[k],
                              decision_function_shape=decision_function_shape[m])

                # Fit SVM to the training data
                clf.fit(X_training, y_training)

                # make the SVM prediction for each test sample and start computing its accuracy
                # hint: to iterate over two collections simultaneously, use zip()
                # Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                # to make a prediction do: clf.predict([x_testSample])
                # --> add your Python code here
                correct_prediction = 0
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    SVM_prediction = clf.predict([x_testSample])
                    if y_testSample == SVM_prediction:
                        correct_prediction += 1

                # check if the calculated accuracy is higher than the previously one calculated. If so, update the
                # highest accuracy and print it together
                # with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2,
                # kernel= poly, decision_function_shape = 'ovo'"
                # --> add your Python code here
                if i == 0 and j == 0 and k == 0 and m == 0:
                    print("-------------------------------------------------------------------------------------------"
                          "--------------------------------------")
                current_accuracy = correct_prediction / len(X_test)
                if highest_accuracy < current_accuracy:
                    highest_accuracy = current_accuracy
                    print("HIGHEST SVM ACCURACY SO FAR: " + str(highest_accuracy) + ", PARAMETERS: C = " + str(c[i]) +
                          ", DEGREE = " + str(degree[j]) + ", KERNEL = " + str(kernel[k]) +
                          ", DECISION_FUNCTION_SHAPE = " + str(decision_function_shape[m]))
                    print("-------------------------------------------------------------------------------------------"
                          "--------------------------------------")
#######################################################################################################################
print("\n** DONE, LAST STATED ACCURACY IS THE HIGHEST FOUND USING EVERY COMBINATION OF THE HYPERPARAMETERS. **")
