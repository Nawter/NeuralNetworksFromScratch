# Part 4: Numerical Gradient Checking

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
## --------------------- Part 4 --------------------
# Helper methods to interact with other methods
    # Get W1 and W2 unrolled into vector:
    def getParams(self):
        parameters = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return parameters

    # Set W1 and W2 using single parameter vector
    def setParams(self, parameters):
        w1Start = 0
        w1End = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(parameters[w1Start:w1End], (self.inputLayerSize, self.hiddenLayerSize))
        w2End = w1End + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(parameters[w1End:w2End], (self.hiddenLayerSize, self.outputLayerSize))


    def computeGradients(self, I, j):
        dJdW1, dJdW2 = self.costFunctionPrime(I, j)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


def computeNumericalGradient(N, I, j):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        #Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(I, j)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(I, j)

        #Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

        #Return the value we changed to zeros
        perturb[p] = 0

    #Return params to original value
    N.setParams(paramsInitial)

    return numgrad
