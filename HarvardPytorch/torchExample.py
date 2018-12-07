import torch
import numpy as np
# x = torch.tensor([5.5, 3])
#
# print(x)
#
# x = x.new_ones(5, 3, dtype = torch.double)
#
# print(x)
#
# x = torch.randn_like(x, dtype = torch.float)
#
# print(x.size())

# x = torch.randn(5,3, dtype = torch.float)
#
# y = torch.randn(5,3, dtype = torch.float)
#
# print(x+y)
#
#
# print(x)
# print(x.add_(y))
#
# print(x)

# a = np.ones(5)
# b = torch.from_numpy(a)
#
# print(a, b)

# x = torch.ones(2, 2, requires_grad = True)
#
# y = x + 2
#
# # print(y)
#
# z = y * y * 3
#
# out = z.mean()
# #
# print(z, out)
#
# # print(y.grad_fn)

# a = torch.randn(2, 2)
# a = ((a*3)/(a-1))
#
# # print(a.requires_grad)
# #
# # a.requires_grad_(True)
# #
# # print(a.requires_grad)
#
# b = (a * a).sum()
#
# print(b.grad_fn)

# out.backward()
#
# print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2

while y.data.norm() < 1000:
    y = y * 2

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)

print(x.requires_grad)
print((x **2))

y.backward(gradients)

print(x.grad)

with torch.no_grad():
    print((x **2).requires_grad)