#%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural net properties (1 output neuron)
input_neuron = 12
hidden_neuron = 10
output_neuron = 10
iteration = 100
cost = np.array([[0],[0]])

# The aim of this NN is to recognize the number 1, drawn with a 4x3 matrix
class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.size, hidden_neuron)
        self.weights2 = np.random.rand(hidden_neuron, output_neuron)
        self.y = y
        self.output = 0

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input.T, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2).T)

    def backprop(self):
        # Cost function C = (output - input)²
        # We have to calculate the derivative of L with respect to the weights
        self.d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)).T)
        self.d_weights1 = np.dot(self.input,  (np.dot(2*((self.y - self.output) * sigmoid_derivative(self.output)).T, self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += self.d_weights1
        self.weights2 += self.d_weights2


# Launch digit recognition
if __name__ == "__main__":
    w = 255
    X1 = np.array([[0], [w], [0],
                   [w], [w], [0],
                   [0], [w], [0],
                   [w], [w], [w]])

    y1 = np.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

    X4 = np.array([[0], [w], [0],
                   [w], [w], [0],
                   [w], [w], [w],
                   [0], [w], [0]])

    y4 = np.array([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]])
    nn = NeuralNetwork(X4,y4)

    for i in range(iteration):
        nn.feedforward()
        nn.backprop()
        
        err = nn.y - nn.output
        c = np.sum(err * err)
        if i == 0:
            cost = np.array([[0],[c]])
        else:
            c_list = np.array([[i],[c]])
            cost = np.concatenate((cost, c_list), axis=1)
        
# Plot & print all variables of the NN

plt.plot(cost[0], cost[1])
plt.show()

#print(nn.input)
#print(nn.weights1)
#print(nn.layer1)
#print(nn.weights2)
#print(nn.d_weights1)
#print(nn.d_weights2)
#print(nn.output)
#print(cost.T)

