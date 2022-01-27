import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(1, 2)
		self.fc2 = nn.Linear(2, 1)

	def forward(self, x):
		x = self.fc2(self.fc1(x))
		return x

criterion = nn.MSELoss()

data = [(1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18)]

net = Net().cuda()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

for epoch in range(100):
	for i, data2 in enumerate(data):
		x, y = iter(data2)
		x, y = Variable(torch.FloatTensor([x]), requires_grad=True).cuda(), Variable(torch.FloatTensor([y]), requires_grad=False).cuda()
		optimizer.zero_grad()
		y_pred = net(x)
		loss = criterion(y_pred, y)
		loss.backward()
		optimizer.step()
	if (epoch % 10 == 0):
		print(f"Epoch {epoch} - loss: {loss}")

# making prediction
print(f"prediction: {net(Variable(torch.Tensor([[[15]]])).cuda())}")	# 1x1x1 tensor