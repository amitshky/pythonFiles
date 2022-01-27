import torch
import torch.optim as optim


t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])	# temperature in celsius
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])	#temperature in unkown unit
t_un = t_u * 0.1

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

print(train_indices, val_indices)

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

def model(t_u, w, b):
	return w * t_u + b

def loss_fn(t_p, t_c):
	squared_diffs = (t_p - t_c) ** 2
	return squared_diffs.mean()

def training_loop (n_epochs, learning_rate, params, t_u, t_c):
	for epoch in range(1, n_epochs + 1):
		if params.grad is not None:
			params.grad.zero()
		t_p = model(t_u, *params)
		loss = loss_fn(t_p, t_c)

		params = (params - learning_rate * params.grad).detach().requires_grad_()
		if epoch % 500 == 0:
			print(f'Epoch: {epoch}	 Loss: {float(loss)}')
	return params


# requires_grad() =  pytorch can track the entire family tree of tensors resulting 
# from operations on params

#training_loop(n_epochs=5000, learning_rate=1e-2, params=torch.tensor([1.0, 0.0], requires_grad=True), t_u=t_un, t_c=t_c)


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)	# params should typically have requires_grad set to True

for epoch in range(5000):
	train_t_p = model(train_t_u, *params)
	train_loss = loss_fn(train_t_p, train_t_c)

	with torch.no_grad():
		val_t_p = model(val_t_u, *params)
		val_loss = loss_fn(val_t_p, val_t_c)
		assert val_loss.requires_grad == False

	optimizer.zero_grad()
	train_loss.backward()
	optimizer.step()

	if epoch % 100 == 0:
		print('Epoch {}, Training loss {}, Validation loss {}'.format(epoch, float(train_loss), float(val_loss)))

print(params)


# if the training loss and the validation loss diverge then you are 
# overfitting the model
# to fix this:
# get enough data
# or scale the model down until it stops overfitting


