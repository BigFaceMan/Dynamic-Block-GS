'''
Author: ssp
Date: 2024-11-18 09:08:20
LastEditTime: 2024-11-18 10:14:36
'''
import torch
import numpy as np

"""
2x = 8
"""
x = torch.tensor([3.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)
tag = 8.0
y = x * 2

for idx in range(100):
    optimizer.zero_grad()
    loss = (2 * y - tag) ** 2
    loss += (y - 4) ** 2
    # loss = 2 * x - tag
    loss.backward()
    print("grad_x : ", x.grad)
    optimizer.step()
    break

