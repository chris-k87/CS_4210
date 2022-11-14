# -------------------------------------------------------------------------
# AUTHOR: CHRISTOPHER KOEPKE
# FILENAME: deep_learning.py
# SPECIFICATION: This program is designed to perform a grid search to find the highest calculated accuracy of a neural
#                   network with the hyperparameters: Hidden layers, Neurons, and learning rate.
#                   The neural network is to be trained to predict 10 possible "fashion icons." The dataset contains
#                   28x28 grayscale images of these icons from the Fashion MNIST dataset.
#                   The model will be built in the function build_model() and return the model built. This function is
#                   called from within the inner loop of the main program body, within which will perform the grid
#                   search function with the outer loop iterating through the number of hidden layers {2, 5, 10}, the
#                   middle loop iterating through the number of neurons {10, 50, 100} and the inner loop iterating
#                   through the learning rate {0.01, 0.05, 0.1}. Within the inner loop, the current accuracy of the
#                   model is calculated and will be printed to the console, include the hyperparameters of that given
#                   iteration if the previous highest calculated accuracy is less than the current(except for the first
#                   interation of the loops, in which the highest accuracy is the current accuracy).
#                   After all loops have executed their iterations, the highest accuracy found will be printed again,
#                   along with the given hyperparameters. The program will also print the model's weights and biases,
#                   the model's summary and learning curve.
# NOTE: I COULD NOT GET THE PLOTS TO BE DISPLAYED, I HAD ISSUES INSTALLING Pydot and GraphViz
#
# FOR: CS 4210.01 - Assignment #4
# TIME SPENT: Took ~2 hours to complete the code, test, and align my various print functions.
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU CAN USE ANY PYTHON LIBRARY TO COMPLETE YOUR CODE.

# importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    # -->add your Python code here

    # Creating the Neural Network using the Sequential API
    # model = keras.models.Sequential()
    # model.add(keras.layers.Flatten(input_shape=[28, 28]))                                # input layer
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))

    # iterate over the number of hidden layers to create the hidden layers:
    # model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))    # hidden layer with ReLU activation function
    for index in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))

    # output layer
    # model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))  # output layer with one neural for each
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))    # class and the softmax activation function
                                                                             # since the classes are exclusive

    # defining the learning rate
    # opt = keras.optimizers.SGD(learning_rate)
    opt = keras.optimizers.SGD(learning_rate)

    # Compiling the Model specifying the loss function and the optimizer to use.
    # model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # return model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model
# END build_model()

# To install TensorFlow on your terminal
# python -m pip install --upgrade tensorflow


# Using Keras to Load the Dataset. Every image is represented as a 28×28 array rather than a 1D array of size 784.
# Moreover, the pixel intensities are represented as integers (from 0 to 255) rather than floats (from 0.0 to 255.0).
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# creating a validation set and scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# For Fashion MNIST, we need the list of class names to know what we are dealing with.
# For instance, class_names[y_train[0]] = 'Coat'
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Iterate here over number of hidden layers, number of neurons in each hidden layer and the learning rate.
# -->add your Python code here
n_hidden = [2, 5, 10]
n_neurons = [10, 50, 100]
l_rate = [0.01, 0.05, 0.1]
correct_prediction = 0
num_of_test = len(X_test)
current_accuracy = 0.0
highest_accuracy = 0.0
hidden_num = 0
neurons_num = 0
rate_num = 0.0

for hid in n_hidden:                          # looking or the best parameters w.r.t the number of hidden layers
    for neu in n_neurons:                     # looking or the best parameters w.r.t the number of neurons
        for lea in l_rate:                    # looking or the best parameters w.r.t the learning rate
            # build the model for each combination by calling the function:
            model = build_model(hid, neu, 10, lea)

            # To train the model;
            # epochs = number times that the learning algorithm will work through the entire training dataset.
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

            # Calculate the accuracy of this neural network and store its value if it is the highest so far.
            # To make a prediction, do:
            class_predicted = np.argmax(model.predict(X_test), axis=-1)

            # -->add your Python code here
            for idx in range(len(class_predicted)):
                if y_test[idx] == class_predicted[idx]:
                    correct_prediction += 1

            # calculate current accuracy of each loop iteration and print if highest along with the hyperparameters
            current_accuracy = correct_prediction / num_of_test
            if hid == 0:
                highest_accuracy = current_accuracy
                print("Highest accuracy so far: " + str(highest_accuracy))
                print("Parameters: " + "Number of Hidden Layers: " + str(hid) + ",number of neurons: " +
                      str(neu) + ",learning rate: " + str(lea))
                print()
            elif highest_accuracy < current_accuracy:
                highest_accuracy = current_accuracy
                hidden_num = hid    # recording the three hyperparameters used to print the final highest accuracy
                neurons_num = neu
                rate_num = lea
                print("Highest accuracy so far: " + str(highest_accuracy))
                print("Parameters: " + "Number of Hidden Layers: " + str(hid) + ",number of neurons: " + str(neu) +
                      ",learning rate: " + str(lea))
                print()

            # reset the variables for recalculation of accuracy
            correct_prediction = 0
            current_accuracy = 0.0

        # END OF INNER LOOP
    # END OF MIDDLE LOOP
# END OF OUTER LOOP

# final print for highest accuracy found and the hyperparameters used
print("\nHIGHEST ACCURACY FOUND AFTER ALL ITERATIONS: " + str(highest_accuracy))
print("PARAMETERS; Number of Hidden Layers: " + str(hidden_num) + ", Number of Neurons: " + str(neurons_num) +
      ", Learning Rate: " + str(rate_num))

# After generating all neural networks, print the final weights and biases of the best model
print("===============================================================================================================")
weights, biases = model.layers[1].get_weights()
print(weights)
print(biases)

# The model’s summary() method displays all the model’s layers, including each layer’s name (which is automatically
# generated unless you set it when creating the layer), its output shape (None means the batch size can be anything),
# and its number of parameters. Note that Dense layers often have a lot of parameters. This gives the model quite a lot
# of flexibility to fit the training data, but it also means that the model runs the risk of overfitting, especially
# when you do not have a lot of training data.
print(model.summary())
img_file = './model_arch.png'
tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)

# plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()
