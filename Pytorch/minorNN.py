import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(3, 6)
		self.fc2 = nn.Linear(6, 6)
		self.fc3 = nn.Linear(6, 6)
		self.fc4 = nn.Linear(6, 1)
		
	def forward(self, x):
		x = F.sigmoid(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		x = F.sigmoid(self.fc3(x))
		x = F.tanh(self.fc4(x))
		
		return x

data = [(0.1, 0.5, 1.0, 1.0), (0.8, 0.5, 0.15, -1.0)]

net = Net()

optimizer = optim.Adam(net.parameters(), lr=1e-3)
x = [0.1, 0.5, 1.0, 1.0]
input = Variable(torch.FloatTensor([x]), requires_grad=True)
output = net(input)
print(output)
# for epoch in range(200):
# 	for i, data2 in enumerate(data):
# 		x, y = iter(data2)
# 		x, y = Variable(torch.FloatTensor([x]), requires_grad=True), Variable(torch.FloatTensor([y]), requires_grad=False)
# 		optimizer.zero_grad()
# 		outputs = net(x)
# 		loss = criterion(outputs, y)
# 		loss.backward()
# 		optimizer.step()
# 	if (epoch % 10 == 0):
# 		print(f"Epoch {epoch} - loss: {loss.data[0]}")
