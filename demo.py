import numpy as np
import matplotlib.pyplot as plt
from nn.model import NeuralNetwork

def generate_circles(n_samples=1000, noise=0.05, factor=0.5):
    """
    Generate bullseye dataset.
    n_samples: total points
    noise: Gaussian noise on radius
    factor: ratio of inner circle radius to outer
    """
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer circle with radius 1
    theta_outer = 2 * np.pi * np.random.rand(n_outer)
    r_outer = 1 + noise * np.random.randn(n_outer)
    x_outer = np.stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)], axis=1)
    y_outer = np.zeros(n_outer)

    # Inner circle with radius = factor
    theta_inner = 2 * np.pi * np.random.rand(n_inner)
    r_inner = factor + noise * np.random.randn(n_inner)
    x_inner = np.stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)], axis=1)
    y_inner = np.ones(n_inner)

    X = np.vstack([x_outer, x_inner])
    Y = np.concatenate([y_outer, y_inner])
    return X, Y


def plot_boundary(nn, X, Y):
    # Create prediction grid
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([nn.predict(p)["result"] for p in grid])
    Z = preds.reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
    plt.scatter(X[:,0], X[:,1], c=Y, cmap="coolwarm", edgecolors="k", s=25)
    plt.title("Bullseye Classification with Custom Neural Network")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("circles_decision_boundary.png", dpi=150)
    plt.show()


def demo_circles():
    print("Generating bullseye dataset...")
    X, Y = generate_circles(n_samples=1200, noise=0.07, factor=0.45)

    print("Initializing neural network...")
    nn = NeuralNetwork(
        layers=[2, 8, 8, 1],
        lr=0.05,
        batch_size=32
    )

    print("Training model...")
    nn.fit(X, Y, epochs=1500)

    print("Plotting decision boundary...")
    plot_boundary(nn, X, Y)



if __name__ == "__main__":
    demo_circles()
