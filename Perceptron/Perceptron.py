#Stephen Reilly 201527474
# Computational Intelligence - COMP 575
# Coursework part 1 - Perceptron

# Import libraries needed for program
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# Sets numpy to only display 3 decimal points for easier reading of values
np.set_printoptions(precision=3, suppress=True)
# Perceptron Class
class Perceptron():
    """
    Perceptron Binary Classifier

    Parameters
    ----------
    learning_rate : int
        learning rate used in update rule, default = 0.01
    iters: int
        Number of iterations to make over data, default = 100
    random_seed: int
        Set random seed for data reproducability, default = 42

    """

    def __init__(self, learning_rate = 0.01,iters = 100, random_seed = 42):
        self.learning_rate = learning_rate
        self.iters = iters
        self.random_seed = random_seed

    def fit(self, X, y):
        """
        Fits the training data, updating the weights

        :param X: np.array - input data
        :param y: np.array - labels
        :return: self
        """
        np.random.seed(self.random_seed)

        self.weights = np.zeros(X.shape[1]+1) #Initialise weights as 0s with an extra weight for the bias

        for i in range(self.iters):
            for data,label in zip(X,y): # combine input data and labels
                pred = self.predict(data) # Get activation score
                update = self.learning_rate * (label - pred) # Find update value using update equation
                self.weights[0] = self.weights[0] + update #apply update to bias
                self.weights[1:] = self.weights[1:] + update * data #apply update to weights

                # If update value is greater than 0, prints the updated weight value to show learning
                if update > 0:
                    print('Epoch', i, 'Weight Updates:')
                    print(self.weights)
        return self


    def predict(self, X):
        """"
        Computes the activation score for the data

        Parameters
        -----------
        X : np.array
            input data

        Returns
        -------
        activation score as np.array
        """
        return np.sign(np.dot(X,self.weights[1:]) + self.weights[0])



# Experiments

#Iris dataset

# Acquire data from sklearn.datasets
iris = datasets.load_iris()

#Split data into data and labels
iris_dataset = iris.data
iris_labels = iris.target

#Split data into first two classes,select 2 features and assign labels that would work with perceptron
set_versi = iris_dataset[0:100, [0, 2]]
set_versi_labels = iris_labels[0:100]
set_versi_labels = np.where(set_versi_labels == 0,1,-1)

# Split data into train and test batches
X_train, X_test, y_train, y_test = train_test_split(set_versi, set_versi_labels, test_size=0.2, random_state=42)


print('\nIris Dataset\n')
iris_ppn = Perceptron() #Create Perceptron instance for training
iris_ppn.fit(X_train, y_train) # Use training data to train weights
predictions = iris_ppn.predict(X_test) #using trained weights to predict classification using test data
accuracy = metrics.classification_report(y_test,predictions) #Get classification report on test data
print('\nSetosa vs Versicolor Length Results:\n',accuracy)

# #Create decision boundary
# x = np.linspace(4,8,100)
# def f(x):
#     y_list = []
#     for i in x:
#         y= (-(iris_ppn.weights[0]/iris_ppn.weights[2])/(iris_ppn.weights[0]/iris_ppn.weights[1]))* i + (-iris_ppn.weights[0]/iris_ppn.weights[2])
#         y_list.append(y)
#     return y_list
# boundary_list = f(x)
#
# #Plot Results
# plt.scatter(set_versi[0:50,0],set_versi[0:50,1], color='blue', label = 'Setosa',alpha=0.4)
# plt.scatter(set_versi[50:100,0],set_versi[50:100,1], color='red', label = 'Versicolor',alpha=0.4)
# plt.plot(x,boundary_list)
# plt.xlabel('Sepal Length')
# plt.ylabel('Petal Length')
# plt.title('Setosa vs Versicolor\nSepal Length vs Petal Length')
# plt.legend()
# plt.tight_layout()
# plt.savefig('set vs ver length boundary')
# plt.show()

# These results show the plot for petal vs sepal length to get the results for the width you will need to change
# the feature selection in line 88 to set_versi = iris_dataset[0:100, [1, 3]] and the range in line  104 to
# x = np.linspace(1,5,100).

# Breast Cancer Dataset

# The breast cancer data requires many iterations to get successful results, these reusults have been included in the
# report however when left the weight updates dominate the terminal so it has been commented out to allow the other data
# to show. To run uncomment all the code below once ( this should keep the comments as comments and make the code
# active.).

# #Acquire breast cancer dataset from sklearn
# cancer = datasets.load_breast_cancer()
#
# print('\nBreast Cancer Dataset\n')
# #split data into data and labels, assign labels to values that will work with perceptron
# cancer_dataset = cancer.data
# cancer_labels = cancer.target
# cancer_labels = np.where(cancer_labels == 0, 1,-1)
#
# # Split into test train batches
# X_train, X_test,y_train, y_test = train_test_split(cancer_dataset,cancer_labels,test_size=0.2,random_state=42)
#
#
# cancer_ppn = Perceptron(iters=1000) #Create instance for training
# cancer_ppn.fit(X_train,y_train) # Train weights using training data
# prediction = cancer_ppn.predict(X_test) # Predict classification using test data
# accuracy = metrics.classification_report(y_test,prediction) #Get classification report based off of predictions
# print('\nBreast Cancer Results:\n',accuracy)
