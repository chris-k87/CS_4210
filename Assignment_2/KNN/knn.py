# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: knn.py
# SPECIFICATION: This program was designed to implement an Instance Based Learning ( 1NN ) and calculate its error rate.
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# --------------------------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
X = []
Y = []
# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# initialize a variable to keep track of wrong predictions
Num_of_wrong_predictions = 0

# loop your data to allow each instance to be your test set
for i, instance in enumerate(db):
    # add the training features to the 2D array X and remove the instance that will be used for testing in this
    # iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning messages

    # transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the
    # instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert values to
    # float to avoid warning messages

    # --> add your Python code here
    X = []
    Y = []
    testSample = []

    # Convert data to floating values and assign The (X,Y) pair to the X array and transform the Labels ('-', '+')
    # into 1.0 or 2.0, respectively and assign the values to the Y array
    for data in db:
        X.append([float(data[0]), float(data[1])])
        if data[2] == '-':
            Y.append(1.0)
        elif data[2] == '+':
            Y.append(2.0)

    # Storing the values of the instance that will be used for testing and assigning them to the testSample array
    X_1 = X[i][0]
    X_2 = X[i][1]
    Y_1 = Y[i]
    testSample.append([X_1, X_2])

    # Removing the instance that will be used for testing
    X.pop(i)
    Y.pop(i)

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)
 
    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]
    # --> add your Python code here
    class_predicted = clf.predict(testSample)

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    # --> add your Python code here
    if not class_predicted == Y_1:
        Num_of_wrong_predictions += 1

# print the error rate
# --> add your Python code here
print('Error Rate: ', Num_of_wrong_predictions/10)






