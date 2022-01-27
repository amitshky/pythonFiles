
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#single neuron
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(1,1)
	
	def forward(self, x):
		x = self.fc1(x)
		return x


def criterion(out, label):
	return (label - out)**2

data = [(1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18)]

net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

for epoch in range(200):
	for i, data2 in enumerate(data):
		x, y = iter(data2)
		x, y = Variable(torch.FloatTensor([x]), requires_grad=True), Variable(torch.FloatTensor([y]), requires_grad=False)
		optimizer.zero_grad()
		outputs = net(x)
		loss = criterion(outputs, y)
		loss.backward()
		optimizer.step()
	if (epoch % 10 == 0):
		print(f"Epoch {epoch} - loss: {loss.data[0]}")

# making prediction
print(f"prediction: {net(Variable(torch.Tensor([[[15]]])))}")	# 1x1x1 tensor