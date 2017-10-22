# Part 3: Backpropagation

## --------------------- Part 1 --------------------
import numpy as np

# I = (hours sleeping, hours studying) , j = Score on test
I = np.array(([3,5], [5,1] , [10,2]), dtype=float)
j = np.array(([75], [82], [93]), dtype=float)

# Normalize
I = I / np.amax(I, axis=0)
j = j/100 #Max test score is 100

## --------------------- Part 2 --------------------
class NeuralNetwork(object):
    def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        #Weights(parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, I):
        #Propagate inputs through network
        self.z2 = np.dot(I, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        jHat = self.sigmoid(self.z3)
        return jHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
## --------------------- Part 3 --------------------
    #Gradient of sigmoid
    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    #Compute cost for given I,j, use weights already stored in class.
    def costFunction(self, I, j):
        self.jHat = self.forward(I)
        J = 0.5*sum((j-self.jHat)**2)
        return J

    #Compute derivative with respect to W1 and W2 for a given I and j:
    def costFunctionPrime(self, I, j):
        self.jHat = self.forward(I)
        delta3 = np.multiply(-(j-self.jHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(I.T, delta2)

        return dJdW1, dJdW2
