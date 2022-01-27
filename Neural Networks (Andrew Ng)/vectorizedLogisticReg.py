import numpy as np 

m = 4	# batch size
n_x = 2	# input nodes
o_y = 1	# output nodes
lr = 0.01	# learning rate

def sigmoid(z):
	return 1 / (1 + np.exp(-(z)))

def d_sigmoid(z):
	return sigmoid(z) * (1 - sigmoid(z))

# Initialization
w = np.random.randn(o_y, n_x)	# weights
b = np.random.randn(1, 1)		# bias
x = np.random.randn(n_x, m)	# input
y = np.random.randint(0, 2, (o_y, m))	# target

dz = np.zeros((o_y, m))
dw = np.zeros((n_x, o_y))
db = np.zeros((1, 1))
cost = 0

print("Initialization")
print(f"Weights(W) = \n{w}")
print(f"Bias(b) = \n{b}")
print(f"Input(X) = \n{x}")
print(f"Target(Y) = \n{y}\n")

	# Forward Propagation
z = np.dot(w, x) + b		# output
a = sigmoid(z)		# activated output
cost += -1 / m * (y * np.log(a) + (1 - y) * np.log(1 - a))	# loss function

# Gradient Descent
dz = a - y
dw = 1 / m * np.dot(x, dz.T)
db = 1 / m * np.sum(dz, axis=1, keepdims=True)

# Update
w = w - lr * dw.T
b = b - lr * db

# Result
print("Result")
print(f"Output(Z) = \n{z}")
print(f"Activated Output(A) = \n{a}")
print(f"Cost = \n{cost}\n")

print(f"dZ = \n{dz}")
print(f"dW = \n{dw}")
print(f"db = \n{db}\n")

print(f"Weights(W) = \n{w}")
print(f"Bias(b) = \n{b}\n")

