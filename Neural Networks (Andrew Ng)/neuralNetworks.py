# A Neural network with 2 layers
# implements the forward and backprop

'''
* Maths behind Neural networks (vectorized)
* Forward prop
* * Z = W * X + b
* * A = sigmoid(Z)
*
* Backprop
* * dZ = A - target
* * dW = 1/m * X * transpose(dZ)
* * db = 1/m * sum(dZ)
*
* Update parameters
* * W := W - lr * dW
* * b := b - lr * db
*
* input   mat dim  => N x BatchSize
* weights mat dim  => NumOfNodesIn(L)thlayer x NumOfNodesIn(L-1)thlayer
* biases  mat dim  => NumOfNodesIn(L)thlayer x 1
'''

import numpy as np


def forward(weights, x, biases):
	return np.add(np.dot(weights, x), biases)

def sigmoid(x):
	return 1 / np.add(1, np.exp(-x))

def d_sigmoid(x): # derivative of sigmoid function
	return sigmoid(x) * (1- sigmoid(x))

# loss function (output, target)
def binary_cross_entropy(output, target):
	return -(target * np.log(output) + (1 - target) * np.log(1 - output))


# AND gate input and target
x     = np.array([[0, 1, 0, 1],
				  [0, 0, 1, 1]])
target = np.array([0, 0, 0, 1])

# parameters and hyperparameters
m  = x.shape[1] # batch size
n0 = x.shape[0] # no. of input neurons
n1 = 4          # no. of neurons in the hidden layer
n2 = 1          # no. of output neurons
lr = 0.1        # learning rate

# randomize weights and biases
w1 = np.random.randn(n1, n0)
w2 = np.random.randn(n2, n1)
b1 = np.zeros((n1,  1))
b2 = np.zeros((n2,  1))

print(f"Input  = \n{x}")
print(f"Target = \n{target}")

for epoch in range(9000):
	# Forward Propagation
	z1   = forward(w1, x, b1)
	a1   = sigmoid(z1)
	z2   = forward(w2, a1, b2)
	a2   = sigmoid(z2)
	loss = binary_cross_entropy(a2, target)
	print(f"Epoch = {epoch + 1}   Output = {np.round(a2, 2)}   Loss = {np.average(loss):.5f}")
	# Gradient Descent
	dz2 = np.subtract(a2, target)
	dw2 = 1/m * np.dot(dz2, a1.T)
	db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
	dz1 = np.dot(w2.T, dz2) * d_sigmoid(z1)
	dw1 = 1/m * np.dot(dz1, x.T)
	db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
	# Update
	w1 -= lr * dw1
	b1 -= lr * db1
	w2 -= lr * dw2
	b2 -= lr * db2

	if np.average(loss) < 0.004:
		break

