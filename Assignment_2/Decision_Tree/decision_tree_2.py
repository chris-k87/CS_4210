# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program is designed to create a decision tree using three different training sets and measure
#                   the accuracy of each model. This process will be done 10 times.
# FOR: CS 4210- Assignment #2
# TIME SPENT: For this specific program, it took me about 8 hours to complete
# -----------------------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. .
# You have to work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
dataset_loop_index = 0

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1,
    # Prepresbyopic = 2, Presbyopic = 3 so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    age = {
        "Young": 1,
        "Prepresbyopic": 2,
        "Presbyopic": 3,
    }
    spectacle = {
        "Myope": 1,
        "Hypermetrope": 2,
    }
    astigmatism = {
        "Yes": 1,
        "No": 2,
    }
    tear = {
        "Normal": 1,
        "Reduced": 2,
    }
    for data in dbTraining:
        X.append([age[data[0]], spectacle[data[1]], astigmatism[data[2]], tear[data[3]]])

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1,
    # No = 2, so Y = [1, 1, 2, 2, ...]
    # --> addd your Python code here
    lenses = {
        "Yes": 1,
        "No": 2,
    }
    for data in dbTraining:
        Y.append(lenses[data[4]])

    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = []
        X_test = []
        Y_test = []
        with open('contact_lens_test.csv', 'r') as csvfile_test:
            reader = csv.reader(csvfile_test)
            for idx, row in enumerate(reader):
                if idx > 0:  # skipping the header
                    dbTest.append(row)

        # transform the features of the test instances to numbers following the same strategy done during training,
        # and then use the decision tree to make the class prediction. For instance:
        # class_predicted = clf.predict([[3, 1, 2, 1]])[0] where [0] is used to get an integer as the predicted
        # class label so that you can compare it with the true label
        # --> add your Python code here
        age = {
            "Young": 1,
            "Prepresbyopic": 2,
            "Presbyopic": 3,
        }
        spectacle = {
            "Myope": 1,
            "Hypermetrope": 2,
        }
        astigmatism = {
            "Yes": 1,
            "No": 2,
        }
        tear = {
            "Normal": 1,
            "Reduced": 2,
        }
        for data in dbTest:
            X_test.append([age[data[0]], spectacle[data[1]], astigmatism[data[2]], tear[data[3]]])

        lenses = {
            "Yes": 1,
            "No": 2,
        }
        for data in dbTest:
            Y_test.append(lenses[data[4]])

        # compare the prediction with the true label (located at data[4]) of the test instance to
        # start calculating the accuracy.
        # --> add your Python code here
        class_predicted = clf.predict(X_test)

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        for idx in range(len(Y_test)):
            if class_predicted[idx] == 1 and Y_test[idx] == 1:
                true_pos += 1
            if class_predicted[idx] == 2 and Y_test[idx] == 2:
                true_neg += 1
            if class_predicted[idx] == 1 and Y_test[idx] == 2:
                false_pos += 1
            if class_predicted[idx] == 2 and Y_test[idx] == 1:
                false_neg += 1
        # print(class_predicted)
        # print('true_pos: ', true_pos)
        # print('true_neg: ', true_neg)
        # print('false_pos: ', false_pos)
        # print('false_neg: ', false_neg)

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        current_accuracy = (true_pos + true_neg)/(true_pos+true_neg+false_pos+false_neg)
        if i == 0:
            lowest_accuracy = current_accuracy
        if i > 0:
            if current_accuracy < lowest_accuracy:
                lowest_accuracy = current_accuracy

    # print the lowest accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here
    print('Final accuracy of ' + ds + ' is: ', lowest_accuracy)
    lowest_accuracy = 0



