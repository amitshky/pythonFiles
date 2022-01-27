import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 

import numpy as np 
import matplotlib.pyplot as plt 

import torchvision.transforms as transforms
import torchvision.datasets

# hyperparameters
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# loading dataset
DATA_PATH = 'datasets'
MODEL_STORE_PATH = 'savedModels\\'

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# to normalize mean and standard deviation must be provided and for MNIST it is 0.1307 and 0.3081

# mnist datasets
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	print("Using GPU!")

class ConvNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)	# kernel_size => 5x5 conv filters
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)					# kernel_size => 2x2 max pool
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

		self.drop_out = nn.Dropout()		# to avoid overfitting of the model
		self.fc1 = nn.Linear(7 * 7 * 64, 1000)
		self.fc2 = nn.Linear(1000, 10)

	def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = x.reshape(x.size(0), -1)		# flattens form 7x7x64 to 3164x1
		x = self.drop_out(x)
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)	# no softmax because CrossEntropyLoss does it for us
		return x


model = ConvNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training loop
total_step = len(train_loader)
loss_list = []
acc_list = []

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

# Testing the model
print("Testing")
total_test = len(test_loader)
model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		inputs = images.to(device)
		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		correct += (predicted.to(device) == labels.to(device)).sum().item()
		total += labels.size(0)

	print(f"Test Accuracy of the model on the 10000 test images: {(correct / total) * 100} %")

torch.save(model.state_dict(), MODEL_STORE_PATH + 'convolutional_nn_model.ckpt')


