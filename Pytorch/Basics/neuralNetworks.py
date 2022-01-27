import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
import matplotlib.pyplot as plt 
from collections import OrderedDict


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)	# adding an extra dimension
t_u = torch.tensor(t_u).unsqueeze(1)

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

# dont use .forward() directly
linear_model = nn.Linear(1, 1)

seq_model = nn.Sequential(OrderedDict([
	('hidden_linear', nn.Linear(1, 8)),
	('hidden_activation', nn.Tanh()),
	('output_linear', nn.Linear(8, 1))
]))

# for name, param in seq_model.named_parameters():
# 	print(name, param.shape)

class SubclassModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(1, 8)
		self.out = nn.Linear(8, 1)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = self.out(x)
		return x


model = SubclassModel()
print(model)
learning_rate = 1e-2
optimizer = optim.Adam(model.parameters(), lr=learning_rate)	# params should typically have requires_grad set to True
loss_fn = nn.MSELoss()

for epoch in range(9000):
	train_t_p = model(train_t_un)
	train_loss = loss_fn(train_t_p, train_t_c)

	with torch.no_grad():
		val_t_p = model(val_t_un)
		val_loss = loss_fn(val_t_p, val_t_c)
		assert val_loss.requires_grad == False

	optimizer.zero_grad()
	train_loss.backward()
	optimizer.step()

	if epoch % 1000 == 0:
		print('Epoch {}, Training loss {}, Validation loss {}'.format(epoch, float(train_loss), float(val_loss)))


print('output', model(val_t_un))
print('answer', val_t_c)


t_range = torch.arange(20.0, 90.0).unsqueeze(1)

#fig = plt.figure(dpi=100)
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), model(0.1 * t_range).detach().numpy(), 'c-')
plt.plot(t_u.numpy(), model(0.1 * t_u).detach().numpy(), 'kx')
plt.show()



