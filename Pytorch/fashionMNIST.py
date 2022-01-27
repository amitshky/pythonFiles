import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 256
EPOCHS = 10

DATA_PATH = 'datasets'
MODEL_STORE_PATH = 'savedModels\\'

mean = 0.2860347330570221
std = 0.3530242443084717

torch.set_printoptions(precision=4, linewidth=120)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans = transforms.Compose([ transforms.ToTensor(), transforms.Normalize( (mean,), (std,)) ])
train_set = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=trans)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
		
		self.fc1 = nn.Linear(12*4*4, 120)
		self.fc2 = nn.Linear(120, 60)
		self.out = nn.Linear(60, 10)
		
	def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = x.reshape(-1, 12*4*4)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.out(x)
# x = torch.softmax(x, dim=1)
		return x


model = ConvNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# model.load_state_dict(torch.load(MODEL_STORE_PATH + 'fashionMNIST.ckpt'))

total_step = len(train_loader)
# loss_list = []
# acc_list = []

print("Training")
for epoch in range(EPOCHS):
	for i, (images, labels) in enumerate(train_loader):
		inputs = images.to(device)
		targets = labels.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		# loss_list.append(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		correct = outputs.argmax(dim=1).eq(targets).sum().item()
		# acc_list.append(correct / BATCH_SIZE)

		if (i + 1) % 100 == 0:
			print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {(correct / BATCH_SIZE) * 100:.2f}%")


print("Training Complete")
torch.save(model.state_dict(), MODEL_STORE_PATH + 'fashionMNIST.ckpt')
