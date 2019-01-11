##%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural net properties
input_neuron = 25           # Number of input neurons, depends on the number of features/pixels to consider
hidden_neuron = 2           # Number of neurons in the hidden layer. Quick rule of thumb (input neurons + output neurons)^0.5 + 5-10
output_neuron = 10          # Number of output neurons, depends on the desired answered
iteration = 30              # Number of iteration for training
cost = np.array([[0],[0]])  # Initializations of the Cost array that we plot at the end

# The aim of this NN is to recognize 0-9 digits
class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x                                                  # Init input variable
        self.weights1 = np.random.rand(self.input.size, hidden_neuron)  # Init of first weights matrix, between input and hidden layers (number of input neurons * number of hidden neurons)
        self.weights2 = np.random.rand(hidden_neuron, output_neuron)    # Init of second weights matrix, between hidden and output layers
        self.y = y                                                      # Init of the true response array 
        self.output = np.zeros(y.shape)                                 # Init of the output array
        self.response = 0                                               # The interpreted response of the NN

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input.T, self.weights1))      # Compute of activation function of hidden neurons
        self.output = sigmoid(np.dot(self.layer1, self.weights2).T)     # Compute of activation of output neurons

    def backprop(self):
        # Cost function C = (output - input)²
        # We have to calculate the derivative of C with respect to each weights matrix
        self.d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)).T)
        self.d_weights1 = np.dot(self.input,  (np.dot(2*((self.y - self.output) * sigmoid_derivative(self.output)).T, self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with new value, whose sign is dependant of the derivative (slope) of the Cost function
        self.weights1 += self.d_weights1
        self.weights2 += self.d_weights2

## Launch digit recognition ##
if __name__ == "__main__":
    w = 255
    X1 = np.array([[0], [0], [w], [0], [0],
                   [0], [w], [w], [0], [0],
                   [w], [0], [w], [0], [0],
                   [0], [0], [w], [0], [0],
                   [0], [0], [w], [0], [0],
                   [w], [w], [w], [w], [w]])

    y1 = np.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

    X4 = np.array([[0], [0], [w], [0], [0],
                   [0], [w], [w], [0], [0],
                   [w], [0], [w], [0], [0],
                   [w], [w], [w], [w], [w],
                   [0], [0], [w], [0], [0],
                   [0], [0], [w], [0], [0]])

    y4 = np.array([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]])

    # Initilization of the neural network with a specific input-output (ex: X1,y1)
    nn = NeuralNetwork(X1,y1)

    # Start FF and BP iterations and store the cost values in a table to plot the result 
    for i in range(iteration):
        nn.feedforward()
        nn.backprop()
        
        # Code for storing the Cost value at iteration i to plot the cost value as a function of the iteration number
        err = nn.y - nn.output                              # The error is the difference between what we want and the output
        c = np.sum(err * err)                               # We compute the Cost array and add each members to obtain a global value
        if i == 0:
            cost = np.array([[0],[c]])                      # Init of the cost array 
        else:
            c_list = np.array([[i],[c]])                    # Temp array listing the cost(i)
            cost = np.concatenate((cost, c_list), axis=1)   # We add the cost(i) array at the end of the global cost array

    # Scan the output array and return the response of the NN if the neuron value is greater than 0.95
    for j in range(nn.output.size):
        if nn.output[j] >= 0.9:
            nn.response = j + 1

        
# Plot cost value as a function of iteration number & print output values of the NN

#print(nn.input)
#print(nn.weights1)
#print(nn.layer1)
#print(nn.weights2)
#print(nn.d_weights1)
#print(nn.d_weights2)
#print(nn.output)
print(nn.response)
#print(cost)

plt.plot(cost[0], cost[1])
plt.show()

