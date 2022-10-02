# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: naive_bayes.py
# SPECIFICATION: This program was designed to implement a Naive-Bayes ML model
# FOR: CS 4210.01 - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -------------------------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# reading the training data in a csv file
# --> add your Python code here
dbTraining = []
dbTest = []
X = []
X_test = []
Y = []
Y_test = []

# reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTraining.append(row)

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
# X =
Outlook = {
    "Sunny": 1,
    "Overcast": 2,
    "Rain": 3,
}
Temperature = {
    "Cool": 1,
    "Mild": 2,
    "Hot" : 3,
}
Humidity = {
    "Normal": 1,
    "High": 2,
}
Wind = {
    "Weak": 1,
    "Strong": 2,
}
for data in dbTraining:
    X.append([Outlook[data[1]], Temperature[data[2]], Humidity[data[3]], Wind[data[4]]])

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
# Y =
PlayTennis = {
    "Yes": 1,
    "No": 2,
}
for data in dbTraining:
    Y.append(PlayTennis[data[5]])


# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
# --> add your Python code here
with open('weather_test.csv', 'r') as csvfile_test:
    reader = csv.reader(csvfile_test)
    for idx, row in enumerate(reader):
        if idx > 0:  # skipping the header
            dbTest.append(row)

# transform the original testing features to numbers and add them to the 4D array X_test.
for data in dbTest:
    X_test.append([Outlook[data[1]], Temperature[data[2]], Humidity[data[3]], Wind[data[4]]])

# printing the header os the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) +
      "PlayTennis".ljust(15) + "Confidence".ljust(15))

# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
# --> add your Python code here
class_predicted = clf.predict_proba(X_test)

for idx in range(len(dbTest)):
    if class_predicted[idx][0] >= 0.75:
        dbTest[idx].pop(5)
        dbTest[idx].append(" YES ")
        print(*dbTest[idx], class_predicted[idx][0])
    elif class_predicted[idx][1] >= 0.75:
        dbTest[idx].pop(5)
        dbTest[idx].append(" NO ")
        print(*dbTest[idx], class_predicted[idx][1])

