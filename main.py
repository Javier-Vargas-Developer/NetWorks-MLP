# main.py

import numpy as np
from network import *


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 1], dtype=float)


network = NetWork([Layer(2, 4), Relu(), Layer(4, 1), Sigmoid()])
network.train(X, Y, epoch=2000, learning_rate=0.4)

print(network.predict(np.array([[1, 1]])))

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure(figsize=(9, 5))
plt.plot(network.losses, color="#3b82f6", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Entrenamiento XOR")
plt.yscale("log")  # opcional, se ve mejor cuando el loss baja mucho
plt.tight_layout()
plt.show()
