import torch
import torch.nn as nn
x = torch.tensor([[[[1,   2,  3],
                   [4,   5,  6],
                   [7,   8,  9],
                   [10,  11, 12],
                   [13,  14, 15]]]])
print(x.permute(1, 2, 0, 3))

