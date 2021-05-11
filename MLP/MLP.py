#Stephen Reilly 201527474
# Computational Intelligence - COMP 575
# Coursework part 2 - MLP

#Imports required for program
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=3, suppress=True)

# Make Moons Dataset (Classification)
# data, labels = datasets.make_moons(100, noise=0.10)
# labels = labels.reshape(100,1)
# plt.scatter(data[:,0], data[:,1], c =labels , cmap=plt.cm.winter)
# plt.show()

# Boston Dateset (Regression)

# Normalise data for better results
scaler = StandardScaler()
dataset = datasets.load_boston()
# Create dataset from boston dataset, reshape labels for use with MLP
data = dataset.data
labels = dataset.target
labels = labels.reshape(labels.shape[0],1)
data = scaler.fit_transform(data)

# Split data into train and test batches
X_train,X_test,y_train,y_test = train_test_split(data,labels, test_size=0.2, random_state=42)

#Set random seed for reproducibility
np.random.seed(42)


#Activation functions for the forward and backwards pass

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def d_relu(x):
    if x <= 0:
        x = 0
    else:
        x = 1
    return x

def linear(x):
    return x

def d_linear(x):
    x = 1
    return x

# Set number of neurons per layer

hidden_layer_neurons = 2
output_layer_neurons = 1

# Initialise starting weights for hidden and output layer between -1 and 1

weights_hidden1 = np.random.uniform(-1,1,(X_train.shape[1], hidden_layer_neurons))
bias_hidden1 = np.random.uniform(-1,1,(weights_hidden1.shape[1], 1))
weights_output = np.random.uniform(-1,1,(weights_hidden1.shape[1], output_layer_neurons))
bias_output = np.random.uniform(-1,1,(weights_output.shape[1], 1))

# Set learning rate for update rule and number of iterations to pass overdata
learning_rate = 0.001
number_of_iterations = 1000


error_list = []
#Begin Training Model
for epoch in range(number_of_iterations):
# Forward pass

    #Input layer into Hidden layer
    hl_1 = np.dot(X_train, weights_hidden1) + bias_hidden1.T #activaction score of input layer with hidden layer weights
    hl_1_out = sigmoid(hl_1) #activation function giving output of hidden layer 1

    #Hidden layer into Output layer
    ol = np.dot(hl_1_out, weights_output)+bias_output.T # activation score of output of hidden layer with output layer weights
    ol_out = linear(ol) #activation function giving final output values


# Backpass

    # Caluclate loss for each predicted output against using MSE
    loss = ((1 / X_train.shape[0]) * ((ol_out - y_train) ** 2))
    # Report back loss average loss for epoch
    print('Loss at Epoch',epoch)
    print('\t', loss.sum())
    error_list.append(loss.sum())
    #Output Layer
    dcost_doutput = ol_out - y_train # Derivative of cost w.r.t outputs
    doutput_dol = d_linear(ol) # Derivative of output w.r.t output activation
    dol_dweightsouput = hl_1_out # Derivative of output activation w.r.t output layer weights
    dcost_weightsoutput = np.dot(dol_dweightsouput.T,dcost_doutput * doutput_dol) # Weight update values
    dcost_biasoutput = dcost_doutput*doutput_dol # Bias update values


    #Hidden Layer
    dcost_dol = dcost_doutput * doutput_dol #Derivative of cost w.r.t output activation
    dol_dhl1output = weights_output # Derivative of output activation w.r.t hidden layer output
    dcost_dhl1output = np.dot(dcost_dol,dol_dhl1output.T) # Derivative of cost w.r.t hidden layer output
    dhl1output_dhl1 =d_sigmoid(hl_1) # Derivative of hidden layer output w.r.t hidden layer activation
    dhl1_dweightshidden1 = X_train # Derivative of hidden layer activation w.r.t hidden layer weights
    dcost_weightshidden1 = np.dot(dhl1_dweightshidden1.T, dhl1output_dhl1*dcost_dhl1output) # Hidden layer weight updates
    dcost_biashidden1 = dhl1output_dhl1*dcost_dhl1output #Hidden layer bias update



    #Update weights

    #Update output layer weights and bias using update rule
    weights_hidden1 -= learning_rate * dcost_weightshidden1
    # Due to shape errors, bias has to be updated in a for loop
    for i in dcost_biashidden1:
        i = i.reshape(bias_hidden1.shape[0], bias_hidden1.shape[1])
        i = learning_rate * i
        bias_hidden1 -= i

    # Update Hidden layer weights and bias using update rule
    weights_output -= learning_rate * dcost_weightsoutput
    #Due to shape errors bias has to be updated in a for loop
    for i in dcost_biasoutput:
        i = i.reshape(bias_output.shape[0], bias_output.shape[1])
        i = learning_rate * i
        bias_output -= i



#Using Trained Model on Test Data
# Forwardpass

#Input layer into Hidden layer
p_1 = np.dot(X_test, weights_hidden1) + bias_hidden1.T #activaction score of input layer with hidden layer weights
p_1_out = sigmoid(p_1) #activation function giving output of hidden layer 1

#Hidden layer into Output layer
p = np.dot(p_1_out, weights_output)+bias_output.T # activation score of output of hidden layer with output layer weights
p_out = linear(p) #activation function giving final output values


# #Used for converting values to their closest classes based on their outputs
# classpreds = []
# for i in p_out:
#     if i >= 0.5:
#         classpreds.append(1)
#     else:
#         classpreds.append(0)
#
# # Create classification report for classification datasets
# classification_acc = metrics.classification_report(y_test,classpreds)
# print(classification_acc)

# # Create explained variance score for regression datasets
regression_acc = metrics.explained_variance_score(y_test,p_out)
print('\nExpected Variance Score using test data: ',regression_acc)

label = 'Learning Rate: ', learning_rate


xl = [*range(1,number_of_iterations+1,1)]


plt.plot(xl,error_list, label = f'Learning Rate: {learning_rate}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.text(xl[-1],50,f'Final loss: {error_list[-1]:.2f}',ha = 'right',va = 'baseline',wrap=True)
plt.title(f'MLP loss with {hidden_layer_neurons} hidden layer neurons and {output_layer_neurons} output layer neuron')
plt.legend()
plt.show()