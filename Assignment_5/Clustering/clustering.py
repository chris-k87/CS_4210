# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: clustering.py
# SPECIFICATION: Implement a Kmeans clustering algorithm and find the 
#                   number of clusters that will produce the highest 
#                   silhouette coefficient for the training data. Plot a 
#                   line graph to visualize the number of clusters to
#                   silhouette coefficients. Finally, print the 
#                   Homogeneity of this kmeans clustering.
# FOR: CS 4210.01 Assignment #5
# TIME SPENT: ~2 hours
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None)  # reading the data by using Pandas library

# assign your training data to X_training feature matrix
X_training = np.array(df.values)
# y_training = np.array(df.values)

max_score = 0.0
maximized_k_value = 0
coefficient_values = []
cluster = []

for k in range(2,20):
    # run kmeans testing different k values from 2 until 20 clusters
    # Use:  kmeans = KMeans(n_clusters=k, random_state=0)
    #      kmeans.fit(X_training)
    # --> add your Python code
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

    # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
    # find which k maximizes the silhouette_coefficient
    # --> add your Python code here
    score = silhouette_score(X_training, kmeans.labels_)
    coefficient_values.append(score)
    cluster.append(k)
    if k == 2:
        max_score = score
    elif max_score < score:
        max_score = score
        maximized_k_value = k
# end of loop

# plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
# --> add your Python code here
plt.xlabel("Clusters")
plt.ylabel("Silhouette Coefficient")
plt.plot(cluster, coefficient_values)
plt.xticks(cluster)
plt.show()

# reading the test data (clusters) by using Pandas library
# --> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None)

# assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
# --> add your Python code here
labels = np.array(df.values).reshape(1, len(df))[0]

# Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
# --> add your Python code here
