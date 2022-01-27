import torch

#a = torch.ones(3)
#print(a)
#a[2] = 2
#print(a)

# points = torch.zeros((5,2,2))
# points[0] = 1.0
# points[1] = 2.0
# points[2] = 3.0
# points[3] = 4.0
# points[4] = 5.0
#print(points)

# passing python list into the constructor
#var = torch.tensor([1,2,3,4,5])
#print(var)

# initializing 2d tensor
#point2d = torch.tensor([[1,2],[2,3],[3,4]])
#print(point2d)
#print(point2d[2,1])
# accessing the storage
#print(point2d.storage())
#print(point2d.storage()[0])

# print("storage offset: ", point2d.storage_offset())
# print("stride: ", point2d.stride())
# print("size: ", point2d.size())

# accessing the 2d tensor = storage_offset + stirde[0] * i + stride[1] * j

# cloning the tensor
#second_point = point2d[1].clone()
#second_point[0] = 9
# print(f"second_point = {second_point}")
# print(f"point2d = {point2d}")

# transpose
# they share the same storage and only differ in shape and stride
#point_t = point2d.t()
# print(f"point2d = {point2d}")
# print(f"point_t = {point_t}")

# transposing multidimensional arrays
# example_tensor = torch.ones(3, 4, 5)
# example_tensor_t = example_tensor.transpose(0, 2)
# print(example_tensor.shape)
# print(example_tensor_t.shape)
# print(example_tensor.stride())
# print(example_tensor_t.stride())

# Numerical types
# double_points = torch.ones(10, 2 , dtype=torch.double)
# short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)

# print(short_points.dtype)
# casting into different dtypes
#double_points = torch.zeros(10,2).to(torch.double)

#indexing

# point = torch.ones(10, 2)
# print(point[1:])	# all rows after first, implicitly all col
# print(point[1:, :])	# all rows after first, all col
# print(point[1:, 0])	# all rows after first, first col

# tensor to numpy array
points = torch.ones(3,4)
# points_np = points.numpy()
# print(points_np)

# numpy array to tensor
# points = torch.from_numpy(points_np)
# print(points)

# serializing tensor (saving and loading)
# torch.save(points, "save.pt")
# # or
# with open("save.pt", "wb") as f:
# 	torch.save(points, f)
# print(points)

# points = torch.load("save.pt")
# # or
# with open("save.pt", "rb") as f:
# 	points = torch.load(f)
# print(points)

# moving tensor to GPUs
# points_gpu = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]], device='cuda')
# # or
# points_gpu = points.to(device='cuda')





