# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: bagging_random_forest.py
# SPECIFICATION: This program is designed to create a single decision tree base classifier by using random forest
#                   algorithms and ensemble classifier. The random forest classifier will be used to recognize digits
#                   and the ensemble classifier will be used to combine the decision trees created by the random
#                   forest algorithm. NOTE: PROGRAM IS INCOMPLETE
# FOR: CS 4210- Assignment #3
# TIME SPENT: Did not complete but spent around 15 hours trying to get it to work.
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
X_test = []
y_training = []
y_test = []
x_testing = []
classVotes = []  # this array will be used to count the votes of each classifier

# reading the training data from a csv file and populate dbTraining
# --> add your Python code here
with open('optdigits.tra', 'r') as training_file:
    reader = csv.reader(training_file)
    for row in reader:
        dbTraining.append(row)

# reading the test data from a csv file and populate dbTest
# --> add your Python code here
with open('optdigits.tes', 'r') as test_file:
    reader = csv.reader(test_file)
    for row in reader:
        dbTest.append(row)

# initializing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
# --> add your Python code here
classVotes.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print("Started my base and ensemble classifier ...")

for k in range(20):  # we will create 20 bootstrap samples here (k = 20).
                     # One classifier will be created for each bootstrap sample

    bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

    # populate the values of X_training and y_training by using the bootstrapSample
    # --> add your Python code here
    for row in range(len(bootstrapSample)):
        X_training.append(bootstrapSample[row][0:-1])
        y_training.append(bootstrapSample[row][-1])

    # fitting the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)  # we will use a single decision tree
                                                                              # without pruning it
    clf = clf.fit(X_training, y_training)
    correct_prediction = 0

    for i, testSample in enumerate(dbTest):
        # make the classifier prediction for each test sample and update the corresponding index value in classVotes.
        # For instance, if your first base classifier predicted 2 for the first test sample, then
        # classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
        # Later, if your second base classifier predicted 3 for the first test sample,
        # then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0] Later, if your third base
        # classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to
        # classVotes[0,0,1,2,0,0,0,0,0,0] this array will consolidate the votes of all classifier for all test samples
        # --> add your Python code here
        prediction = clf.predict([testSample[0:-1]])
        classVotes[0][int(prediction)] += 1

        # for only the first base classifier, compare the prediction with the true label of the test sample here
        # to start calculating its accuracy if k == 0:
        # --> add your Python code here
        if k == 0:
            if int(prediction) == int(dbTest[i][-1]):
                correct_prediction += 1

    if k == 0:  # for only the first base classifier, print its accuracy here
        # --> add your Python code here
        accuracy = correct_prediction / len(dbTest)
        print("Finished my base classifier (fast but relatively low accuracy) ...")
        print("My base classifier accuracy: " + str(accuracy))
        print("")

# now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground
# truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
# --> add your Python code here
correct_prediction = 0
accuracy = 0
print("classVotes: ", classVotes)

final_prediction = clf.predict(dbTest[:][0:-1])
for index in range(len(dbTest)):
    majority_vote = classVotes[index][0]
    for idx in range(10):
        if majority_vote < classVotes[index][idx]:
            majority_vote = classVotes[index][idx]
            final_ensemble_prediction = idx

    print(final_ensemble_prediction)
    if final_ensemble_prediction == int(dbTest[index][-1]):
        correct_prediction += 1

# printing the ensemble accuracy here
accuracy = correct_prediction / 20
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=20)  # this is the number of decision trees that will be generated by
                                            # Random Forest. The sample of the ensemble method used before

# Fit Random Forest to the training data
clf.fit(X_training, y_training)

# make the Random Forest prediction for each test sample. Example:
# class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
# --> add your Python code here
correct_prediction = 0
for i, testSample in enumerate(dbTest):
    class_predicted_rf = clf.predict([testSample[0:-1]])

    # compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    # --> add your Python code here
    if int(prediction) == int(dbTest[i][-1]):
        correct_prediction += 1

accuracy = correct_prediction/len(dbTest)
# printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
