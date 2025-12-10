import numpy as np
import matplotlib.pyplot as plt
'''
My own developed Neural network which takes user input for learning rate and batch size. I chose to 
use BCE aka Binary Cross Entropy loss function, as my endgoal is to integrate this into my nba prediction
pipeline, where I predict between 0 and 1(0 for home loss, 1 for home win).
'''
class NeuralNetwork:
    """
    Simple fully-connected neural network with ReLU hidden layers
    and sigmoid output, trained with binary cross-entropy.
    Supports minibatch gradient descent.
    """
    def __init__(self, layers, lr=0.001, batch_size=1):
        self.layers = layers # As a vector of number of nodes in each layer [input, hidden1, hidden2, ..., output] in example
        self.lr = lr # Initialize learning rate
        self.batch_size = batch_size

        
        weights_arr = [] # First only an empty vector, but we append weight matrices, one for each layer

        #Initialize the weights as random for each weight of every layer
        for i in range(1,len(layers)-1):
            weights_arr.append(np.random.randn(layers[i],layers[i-1]) * np.sqrt(2/(layers[i-1])))
        last = len(layers) - 1
        weights_arr.append(np.random.randn(layers[last],layers[last-1]) * np.sqrt(1/(layers[last-1])))
        self.weights_arr = weights_arr

        #Initialize biases as column for every layer, except input layer
        biases = [np.zeros(layers[layer]).reshape(-1, 1) for layer in range(1, len(layers))]
        self.biases = biases

    # Sigmoid functions for output layer
    def sigmoid(self, x): 
        return (1/(1 + np.exp(-x))) 
    
    def dsigmoid(self, a):
        return a * (1 - a)

    # Relu functions for hidden layers
    def relu(self, x): 
        return np.maximum(0, x)
    
    def drelu(self, a): 
        return (a > 0).astype(float)

    def forward_prop(self, x):
        activations = []
        z = []
        activations.append(x)
        weights_arr = self.weights_arr
        biases = self.biases
        for i in range(len(weights_arr)-1):

            z.append(np.dot(weights_arr[i], activations[i]) +  biases[i]) # Make so that the biases is added into columns of x
            activations.append(self.relu(z[i]))
        # Use sigmoid for output layer
        last = len(weights_arr) - 1
        z.append(np.dot(weights_arr[last], activations[last]) +  biases[last]) 
        activations.append(self.sigmoid(z[last]))
        self.activations = activations
        self.z = z

    def back_prop(self, y):
        activations = self.activations
        weights_arr = self.weights_arr
        z = self.z
        L = len(self.layers)
        dz = [None] * L
        dw = [None] * L
        db = [None] * L

        # Output layer
        dz[L-1] = activations[L-1]-y
        dw[L-1] = np.dot(dz[L-1], activations[L-2].T)/self.batch_size
        db[L-1] = (np.sum(dz[L-1], axis=1, keepdims=True)) / self.batch_size

        # Backprop through hidden layers
        for i in reversed(range(1, L-1)):
            dz[i] = np.dot(weights_arr[i].T, dz[i+1]) * self.drelu(z[i-1])
            dw[i] = np.dot(dz[i], activations[i-1].T)/self.batch_size
            db[i] = np.sum(dz[i], axis=1, keepdims=True) / self.batch_size
        biases = self.biases
        weights_arr = self.weights_arr

        # Make the change of the amount gradient times lr
        for i in range(1, L):
            self.weights_arr[i-1] -= self.lr * dw[i]
            self.biases[i-1] -= self.lr * db[i]

    # Fitting
    def fit(self, X, Y, epochs=100):
        batch_size = self.batch_size
        samples = X.shape[0]
        batches = samples // batch_size
        losses = []

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        for i in range(epochs):
            # Shuffle for every epoch
            indices = np.random.permutation(samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for j in range(batches):
                start = j*batch_size
                end = start + batch_size
                batch = slice(start, end)
                self.forward_prop(X_shuffled[batch].T)
                self.back_prop(Y_shuffled[batch].T)
            self.forward_prop(X.T)

            activations = self.activations[-1]              # final output layer
            eps = 1e-8                                   # gives nan when log(0), so use small number
            a = np.clip(activations, eps, 1 - eps)

            loss = -1/samples * np.sum(Y.T*np.log(a) + (1-Y.T)*np.log(1-a))
            losses.append(loss)
            accuracy = np.mean((a>= 0.5).astype(int) == Y.T)
            print(f"Epoch[{i}]: BCE Loss {loss}, Accuracy {accuracy}")

        plt.plot(losses)
        plt.xlabel("EPOCHS")
        plt.ylabel("Loss value")
    

    # Make prediction
    def predict(self, X):

        # Convert to NumPy array
        X = np.array(X)


        # If X is one sample of shape (n_features,), reshape (1, n_features)
        # If X is already (n_samples, n_features), do nothing
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.forward_prop(X.T)

        probs = self.activations[-1].flatten()       

        # Predicted classes
        predictions = (probs >= 0.5).astype(int)

        # Confidence for each prediction
        confidences = np.where(predictions == 1, probs, 1 - probs)

        # Build output list of dicts
        results = [
            {"result": int(predictions[i]), "confidence": float(confidences[i])}
            for i in range(len(predictions))
        ]

        # when given only one sample, return a single dict
        if len(results) == 1:
            return results[0]

        return results



    