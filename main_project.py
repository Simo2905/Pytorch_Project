import torch
import matplotlib.pyplot as plt

# I create two tensors that I will use as training data for the model

x = torch.tensor([55, 65, 75, 54, 67, 78, 53, 65, 76, 60, 70, 81, 51, 68], dtype = torch.float32).unsqueeze(1) # unsqueeze() adds a dimension to the tensor; otherwise, it generates an error
y = torch.tensor([13, 18, 24, 12, 20, 26, 12, 18, 24, 14, 21, 28, 12, 20], dtype = torch.float32).unsqueeze(1)

plt.scatter(x,y)

