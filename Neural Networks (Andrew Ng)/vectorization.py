import numpy as np 
import time
import math

a = np.random.rand(1000000)
b = np.random.rand(1000000)

start = time.time()
c = np.dot(a, b)
end = time.time()

print(c)
print(f"vectorized version: {1000 * (end - start)}ms")

c = 0
start = time.time()
for i in range(1000000):
	c += a[i] * b[i]
end = time.time()

print(c)
print(f"for loop version: {1000 * (end - start)}ms")

u = np.random.rand(1000000)
start = time.time()
v = np.exp(u)
end = time.time()

print(v)
print(f"vectorized version: {1000 * (end - start)}ms")

v = np.zeros((1000000, 1))
start = time.time()
for i in range(1000000):
	v[i] = math.exp(u[i])
end = time.time()

print(v)
print(f"for loop version: {1000 * (end - start)}ms")
