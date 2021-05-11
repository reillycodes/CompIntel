#Stephen Reilly 201527474
# Computational Intelligence - COMP 575
# Coursework part 3 - Particle Swarm Optimisation

#Import Libraries needed for program
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pyswarms  as ps


#MlP Class
class Layer:
    """
    Create a layer for the GA MlP

    Parameters
    ----------
    W: np.array
        Weight values passed from the GA
    B: np.array
        Bias values passed from the GA
    """
    def __init__(self,W,B):
        self.weights = W
        self.bias = B

    def forward(self, inputs):
        """
        Computes the activation score for the data using dot product of weights + bias

        Parameters
        ----------
        inputs: np.array
            Input data
        Returns
        -------
        output: np.array
            Activation scores
        """
        self.output = np.dot(inputs,self.weights) + self.bias
        return self.output

class Relu:
    """
    Relu activation function

    Parameters
    ----------
    inputs: np.array
        Input data
    Returns
    -------
    output: np.array
        Activation function output values
    """
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Sigmoid:
    """
    Sigmoid Activation function

     Parameters
    ----------
    inputs: np.array
        Input data
    Returns
    -------
    output: np.array
        Activation function output values
    """
    def forward(self, inputs):
        self. output = 1/(1+np.exp(-inputs))

class Linear:
    """
    Linear Activation Function

    Parameters
    ----------
    inputs: np.array
        Input data
    Returns
    -------
    output: np.array
        Activation function output values
    """
    def forward(self,inputs):
        self.output = inputs

#Set seed for reproducibility
np.random.seed(42)

# Acquire and normalise data
scaler = StandardScaler()
dataset = datasets.load_boston()
data = dataset.data
labels = dataset.target
labels = labels.reshape(labels.shape[0],1)
data = scaler.fit_transform(data)

#Split data into test and train batches
X_train,X_test,y_train,y_test = train_test_split(data,labels, test_size=0.2, random_state=42)


def forwardpass(W, X):

    # Input Layer to Hidden Layer
    W1 =W[0:65].reshape((13,5)) # Reshape weights into correct shapes
    B1 = W[65:70].reshape((1,5)) # Reshape bias into correct shape

    layer1 = Layer(W1,B1) #Create Hidden layer and pass in weights from GA
    activation1 = Sigmoid() #Select activation function for hidden layer
    layer1.forward(X) # Run input data through hidden layer  to get activation score
    activation1.forward(layer1.output) # Run activation scores through activation Function

    #Hidden Layer to Output Layer
    W3 = W[70:75].reshape((5,1))# Reshape weights into correct shapes
    B3 = W[-1].reshape((1,1)) # Reshape bias into correct shape

    output_layer = Layer(W3,B3) #Create output layer and pass in weight from GA
    output_activation = Linear() # Select activation function for output layer
    output_layer.forward(activation1.output) # Run hidden layer output through output layer to get activation score
    output_activation.forward(output_layer.output) # Run activation score through activation function to get final outputs

    loss = ((1/X_train.shape[0])*((output_activation.output - y_train)**2)) # Calculate MSE
    loss = loss.sum() # Sum all losses to give us the average loss across all data
    return loss

# Objective function to minimise
def objective_function(pso_weights):
    n_particles = pso_weights.shape[0] #set number of particles equal to number of weights needed
    loss = [forwardpass(pso_weights[i], X_train) for i in range(n_particles)] # Run each particle through the forward pass to calculate fitness
    return np.array(loss)

# Predict function to use on optimal solution
def predict(pos, X):

    W1 = pos[0:65].reshape((13, 5)) # Reshape best solution weights
    B1 = pos[65:70].reshape((1, 5)) # Reshape best solution biases

    layer1 = Layer(W1, B1)
    activation1 = Sigmoid()
    layer1.forward(X)
    activation1.forward(layer1.output)

    W3 = pos[70:75].reshape((5, 1))
    B3 = pos[-1].reshape((1, 1))

    output_layer = Layer(W3, B3)
    output_activation = Linear()
    output_layer.forward(activation1.output)
    output_activation.forward(output_layer.output)

    return output_activation.output # Return predicted outputs of test data

#Paramters used for swarm
options = {'c1': 0.5, 'c2': 0.9, 'w':0.9}
dimensions = 76

#Initalise PSO
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)
#Run PSO on objective function
cost,pos = optimizer.optimize(objective_function,iters=1000)

#Run predict to see how model does with test data
test_prediction  = predict(pos, X_test)
# Get explained variance score to see how well it has performed
explained_variance_score = metrics.explained_variance_score(y_test, test_prediction)
print(explained_variance_score)

