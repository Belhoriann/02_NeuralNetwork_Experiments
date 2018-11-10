import numpy as np    

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Neural net properties (1 output neuron)
input_neuron = 12
hidden_neuron = 4

# The aim of this NN is to recognize the number 1, drawn with a 4x3 matrix
class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], hidden_neuron)
        self.weights2 = np.random.rand(hidden_neuron, 1)
        self.y = y
        self.output = 0

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # Cost function C = (input - output)²
        # We have to calculate the derivative of L with respect to the weights
        self.cost = (self.y - self.output) * (self.y - self.output)
        #d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output) * sigmoid_derivative(self.output))
        #d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        #self.weights1 += d_weights1
        #self.weights2 += d_weights2

if __name__ == "__main__":
    X = np.array([[0], [1], [0],
                  [1], [1], [0],
                  [0], [1], [0],
                  [1], [1], [1]])

    y = np.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(100):
        nn.feedforward()
        #nn.backprop()

#a = nn.input.flatten()
#print(a.reshape(-1,1))
#print(nn.weights1)
#print(nn.layer1)
#print(nn.weights2)
#print(nn.cost)
print(nn.output)
