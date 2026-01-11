# Neural Network From Scratch

A fully custom neural network implementation built using **NumPy only**.

This project was created to deeply understand and reproduce the core mechanics of neural networks, including forward propagation, backpropagation, loss computation, and gradient-based optimization — without relying on high-level deep learning frameworks.

The network was **experimented with as a potential primary model** in a larger NBA game prediction system.  
After evaluation under realistic conditions, it is currently retained as an **experimental and benchmarking model**, rather than the production choice.

---

## Motivation

Modern machine learning frameworks abstract away most of the underlying mechanics of neural networks.

This project exists to:

- Build a first-principles understanding of neural network training
- Implement forward and backward passes explicitly
- Experiment with model behavior under real-world data
- Benchmark a from-scratch neural network against standard ML models
- Create a lightweight, dependency-free implementation for experimentation

---

## Features

- Fully connected feedforward neural network
- ReLU and Sigmoid activation functions
- Binary cross-entropy loss
- Gradient-based training with mini-batch support
- Evaluation utilities:
  - Accuracy
  - Precision
  - Recall
  - F1 score
- Simple, readable, and extensible design

---

## Usage Example (Binary Classification)

```
from nn.model import NeuralNetwork

nn = NeuralNetwork(
    layers=[2, 32, 16, 1],
    lr=0.001,
    batch_size=32
)

nn.fit(X_train, y_train, epochs=200)
predictions = nn.predict(X_test)
```

---

## Demo

The repository includes a demonstration script (`demo.py`) that trains the network on a non-linear binary classification task.

The resulting decision boundary visualization is saved as:

```
circles_decision_boundary.png
```

This demo illustrates the network’s ability to learn non-linear decision boundaries using only NumPy-based operations.

---

## Performance on Historical NBA Game Data

The custom neural network was evaluated against several established machine-learning models using historical NBA regular-season data.

Dataset split:

- Training data:
  - 2015–2016 season through December 31, 2023
- Validation data:
  - All games played on or after January 1, 2024
  - Entire 2024–25 season

This forward-looking split avoids data leakage across seasons and reflects realistic deployment conditions.

---

## Model Comparison


| Model | Parameters | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------------|----------|-----------|--------|----------|---------|
| Logistic Regression | max_iter=1000 | 0.6543 | 0.6477 | 0.7866 | 0.7104 | 0.6430 |
| Random Forest | n_estimators=300, max_depth=12 | 0.6471 | 0.6364 | 0.8064 | 0.7114 | 0.6336 |
| XGBoost | 400 trees, lr=0.03 | 0.6436 | 0.6453 | 0.7526 | 0.6949 | 0.6343 |
| LightGBM | 400 trees, lr=0.03, leaves=31 | 0.6426 | 0.6409 | 0.7668 | 0.6982 | 0.6320 |
| Custom Neural Network | layers=[input, 64, 32, 1], lr=0.001, epochs=40 | 0.6456 | 0.6320 | 0.8206 | 0.7141 | 0.6307 |


---

## Summary

The custom neural network performs competitively with established machine-learning models and achieves the highest recall among the tested approaches.

However, during live and forward-looking evaluation, simpler models demonstrated more stable and reliable performance. As a result, this neural network is currently maintained as an **experimental and research-focused model**, rather than the production default.

This outcome reflects a deliberate engineering decision based on empirical performance rather than model complexity.

---

## Notes

This project is not intended to replace production-grade deep learning frameworks.

Instead, it serves as:

- A first-principles learning exercise
- An experimental benchmark against standard ML models
- A transparent reference implementation of neural network fundamentals
