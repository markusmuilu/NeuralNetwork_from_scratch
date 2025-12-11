# Neural Network From Scratch

A minimal, fully custom neural network implementation built with NumPy.  
This project was created to understand and reproduce the core mechanics of forward propagation, backpropagation, and training using gradient descent.  
The package will also used inside my NBA game prediction project as the primary model, and was built with that purpose in mind.

## Features

- Fully connected feedforward architecture  
- ReLU and Sigmoid activation functions  
- Binary cross-entropy loss  
- Gradient-based training with minibatch support  
- Evaluation utilities (accuracy, precision, recall, F1)  
- Simple and extensible design  

## Usage Example (Binary Classification)

```python
from nn.model import NeuralNetwork

nn = NeuralNetwork(
    layers=[2, 32, 16, 1],
    lr=0.001,
    batch_size=32
)

nn.fit(X_train, y_train, epochs=200)
predictions = nn.predict(X_test)
```

## Demo

The repository includes a demonstration script (`demo.py`) showing how the model adapts to non-linear classification tasks.  
The resulting visualization is saved as `circles_decision_boundary.png`.

You can view the generated plot here:

![Decision Boundary Example](circles_decision_boundary.png)

