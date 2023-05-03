##Importing file and organizing columns

# Importing libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing dataset
s = os.path.join('https://archive.ics.uci.edu', 'ml',
                 'machine-learning-databases',
                 'iris', 'iris.data')

df = pd.read_csv(s, header=None, encoding='utf-8')

# Renaming columns
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

# Mapping output values to int
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Printing out pandas dataframe
df

# Defining input and target variables for both training and testing
X = df.iloc[:100, [0, 1, 2, 3]].values
y = df.iloc[:100, [4]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##Creating artificial neuron and training it with the Iris dataset records

# Importing libraries
import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.

        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

        # The Sigmoid function, an S shaped curve,
        # normalizes the weighted sum of the inputs between 0 and 1.

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

        # The derivative of the Sigmoid function.
        # The gradient of the Sigmoid curve

    def sigmoid_derivative(self, x):
        return x * (1 - x)

        # The training phase adjusts the weights each time to reduce the error

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            # Training inputs are processed
            output = self.think(training_inputs)

            # Calculate the error
            error = training_outputs - output

            # Adjustments refers to the backpropagation process
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustments

    # The neural network predicts new records.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":
    # Assigning the perceptron to an object
    neural_network = NeuralNetwork()

    # Printing synaptic weights before training
    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set. We have 80 examples, each consisting of 4 input values
    # and 1 output value.
training_inputs = X_train
training_outputs = y_train

# Train the neural network using a training set.
# The number of iterations has been set to 1000
neural_network.train(training_inputs, training_outputs, 1000)

# Showing Synaptic weights after training
print("Synaptic weights after training: ")
print(neural_network.synaptic_weights)

# Deploying Neuron on training data
predicted = neural_network.think(X_test)

# Transforming results into Pandas Dataframe
predicted_df = pd.DataFrame({'Result': predicted[:, 0]})


##Create a function to get a precise result from the Artificial Neuron

#Importing libraries
from sklearn.metrics import classification_report

#If the score is higher than 0.5 then it's a 1 otherwise a 0
def getResult(score):
    if score < 0.5:
        return 0
    elif score >= 0.5:
        return 1

#Apply function on predicted dataframe
predicted_df = predicted_df.Result.apply(lambda x: getResult(x))

#Evaluate model performance
print(classification_report(y_test, predicted_df))