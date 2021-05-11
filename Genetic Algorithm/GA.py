#Stephen Reilly 201527474
# Computational Intelligence - COMP 575
# Coursework part 3 - Genetic Algorithm

#Import libraries needed to run program
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

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

# Create and Scale Data
scaler = StandardScaler()
dataset = datasets.load_boston()
data = dataset.data
labels = dataset.target
labels = labels.reshape(labels.shape[0],1)
data = scaler.fit_transform(data)

#Split data into test and train batches
X_train,X_test,y_train,y_test = train_test_split(data,labels, test_size=0.2, random_state=42)

# Set seed for Reproducibility
np.random.seed(42)

# Forward pass function to pass weights and data to get fitness for each chromosome
def forwardpass(W,X):

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

    return output_activation.output

#Objective function the GA is attempting to minimise
def objective_function(ga_weights):

    predicted_output = forwardpass(ga_weights, X_train) # Predicted output of data
    loss = ((1/X_train.shape[0])*((predicted_output - y_train)**2)) # Calculate MSE
    loss = loss.sum() # Sum all losses to give us the fitness for each chromosome
    return loss

#Parameters used within the GA
params = {'max_num_iteration': 500,\
                   'population_size':300,\
                   'mutation_probability':0.3,\
                   'elit_ratio': 0.1,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.1,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':50}

# params = {'max_num_iteration': 1000,\
#                    'population_size':1000,\
#                    'mutation_probability':0.1,\
#                    'elit_ratio': 0.1,\
#                    'crossover_probability': 0.5,\
#                    'parents_portion': 0.3,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':50}

# Set the upper and lower bounds as well as the amount of weights needed for the algorithm
varbound =np.array([[-10,10]]*76)

#Create Model
model = ga(function=objective_function, dimension=76, variable_type='real', variable_boundaries=varbound, algorithm_parameters= params)
#Run Model
model.run()

# Testing best solution

results = model.output_dict #Best Solution found by GA
report = model.report # Report of loss at each generation

test_predicted_outputs = forwardpass(results['variable'], X_test) # Forward pass using best solution weights and test data
explained_variance = metrics.explained_variance_score(y_test, test_predicted_outputs) #Getting explained variance score based off of predicted vs ground truth
print(f'\n\nExplained Varience Score: {explained_variance:.2f}') #Print results
