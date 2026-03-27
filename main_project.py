import torch
import matplotlib.pyplot as plt

# I create two tensors that I will use as training data for the model

x = torch.tensor([55, 65, 75, 54, 67, 78, 53, 65, 76, 60, 70, 81, 51, 68], dtype = torch.float32).unsqueeze(1) # unsqueeze() adds a dimension to the tensor; otherwise, it generates an error
y = torch.tensor([13, 18, 24, 12, 20, 26, 12, 18, 24, 14, 21, 28, 12, 20], dtype = torch.float32).unsqueeze(1)

plt.scatter(x,y)

# =============================================================================================================================
# Let's create our first simple neural network with a single neuron

torch.manual_seed(42)
model = torch.nn.Linear(1,1) # Calculate a simple linear function using weights and bias
                             # Changing W modifies the slope of the line; changing B modifies its position
                             # The numbers represent the number of inputs and outputs; the number of outputs corresponds to the number of neurons
loss = torch.nn.MSELoss()    # I create the loss function to calculate the error; for a regression problem, I use MSE (Mean Squared Error)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1) # After calculating the loss function, we calculate the gradients of each parameter; the larger the gradient, the greater the parameter's impact on the error
model(x)

# =============================================================================================================================
# In this step, we'll train our model

loss = torch.nn.MSELoss() # Instantiate the MSELoss class
for epoch in range(3000): # Let's start with 1000, then we'll switch to more epochs because the error was still decreasing
  optimizer.zero_grad()   # Set all gradients to zero
  y_pred = model(x)
  loss_fn = loss(y_pred, y)
  loss_fn.backward()
  optimizer.step()
  if epoch % 100 == 0:
    print(f"Epoch: {epoch} Loss: {loss_fn}")
  if epoch in [0, 2, 4, 15, 2999]:
    plt.plot(x, y_pred.detach(), label = f"Epoch: {epoch}")

plt.scatter(x,y)
plt.legend()

# =============================================================================================================================

