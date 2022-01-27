import csv
import numpy as np 
import torch 

wine_path = "C:/_My Files/Python/_MachineLearning/Pytorch/WineQuality/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
#print(wineq_numpy)

col_list = next(csv.reader(open(wine_path), delimiter=";"))
#print(wineq_numpy.shape, col_list)

wineq = torch.from_numpy(wineq_numpy)
#print(wineq)
#print(wineq.shape, wineq.type())

data = wineq[:, :-1]	# select all rows and all columns exept the last
#print(data, data.shape)

target = wineq[:, -1].long() 	#select all rows and the last col # long() = treat labels as an integer vector of scores
#print(target, target.shape)

target_onehot = torch.zeros(target.shape[0], 10)
#print(target_onehot)
# scatter_ = to achieve one-hot encoding
# scatter_ fills the tensor with values from a source tensor along the indices provided as arguments
# _  at the end of scatter means it won't return a tensor
# for each row, take the index of the target label, 
# and use it as the column index to set the value 1.0 
# the result is a tensor encoding categorical information
# unsqueeze() addes extra dimensions

#print(target_onehot.scatter_(1, target.unsqueeze(1), 1.0))

target_unsqueeze = target.unsqueeze(1)
#print(target_unsqueeze, target_unsqueeze.shape)

# obtain mean and standard deviation for each col
data_mean = torch.mean(data, dim=0)
#print(data_mean)
data_var = torch.var(data, dim=0)
#print(data_var)

data_normalize = (data - data_mean) / torch.sqrt(data_var)
#print(data_normalize)

# .le = less of equal to
bad_data = data[torch.le(target, 3)]
mid_data = data[(torch.gt(target, 3)) & torch.lt(target, 7)]
good_data = data[torch.gt(target, 7)]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
	print('{:2} {:20} {:10.2f} {:10.2f} {:10.2f}'.format(i, *args))
