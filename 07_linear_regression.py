# 1) Design model (input, output size, forward pass)
# 2) Construct loops and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: calculate gradients

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare data

# make_regression : generate a random (linear?) regression problem
# returns:
# Xndarray of shape (n_samples, n_features)
# The input samples.

# yndarray of shape (n_samples,) or (n_samples, n_targets)
# The output values.

# coefndarray of shape (n_features,) or (n_features, n_targets)
# The coefficient of the underlying linear model. It is returned only if coef is True.

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32)).view(-1,1)

n_samples, n_features = X.shape

# initialize linear model

input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# loss and optimizer
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# traning loop
num_epochs = 100
 
for epoch in range(num_epochs):
    #forward pass
    y_pred = model(X)

    #loss
    l = loss(y, y_pred)

    # backward pass
    l.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {l.item():.4f}')

# plot

predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()