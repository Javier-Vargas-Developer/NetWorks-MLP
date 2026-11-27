import numpy as np

class Layer:
    def __init__(self, n_features, n_neuron, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-1, 1, (n_features, n_neuron))
        self.b = rng.uniform(-1, 1, (n_neuron))

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z

    def backward(self, grad):
        self.grad_W = self.X.T @ grad
        self.grad_b = grad.sum(axis=0)
        return grad @ self.W.T


class Relu:
    def forward(self, X):
        self.Z = X
        return np.maximum(X, 0)

    def backward(self, grad):
        return grad * (self.Z > 0).astype(float)


class Sigmoid:
    def forward(self, X):
        self.A = 1 / (1 + np.exp(-X))
        return self.A

    def backward(self, grad):
        return grad * self.A * (1 - self.A)


class NetWork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self, X, Y, epoch=1000, learning_rate=0.2):
        self.losses = []
        for e in range(epoch):
            pred = self.forward(X)
            error = pred - Y
            loss = (error**2).mean()

            grad = 2 * error / len(X)
            self.backward(grad)

            for layer in self.layers:
                if hasattr(layer, "W"):
                    layer.W -= learning_rate * layer.grad_W
                    layer.b -= learning_rate * layer.grad_b

            self.losses.append(loss)

    def predict(self, X):
        return self.forward(X)
