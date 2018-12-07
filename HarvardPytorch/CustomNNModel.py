import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DynamicNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = nn.Linear(D_in, H)
        self.middle_linear = nn.Linear(H, H)
        self.out_linear = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min = 0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min = 0)
        y_pred = self.out_linear(h_relu)
        return y_pred


# class TwoLayerNet(nn.Module):
#     def __init__(self, D_in, H, D_out):
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = nn.Linear(D_in, H)
#         self.linear2 = nn.Linear(H, D_out)
#
#     def forward(self, x):
#         # h_relu = self.linear1(x).clamp(min = 0)
#         sigmoid = F.sigmoid(self.linear1(x))
#         y_pred = self.linear2(sigmoid)
#         return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)

criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum=0.9)

for i in range(5000):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()