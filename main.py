import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.tensor([1.0], requires_grad=True)
learning_rate = 0.1

optimizer = torch.optim.SGD([x], lr=learning_rate)

for _ in range(100):
    optimizer.zero_grad()
    y = -1 * (x ** 2) + 3 * x
    print(x.item(),y.item())
    (-y).backward()
    optimizer.step()

x_final = x.item()

x_plot = np.linspace(-1, 4, 100)
y_plot = -x_plot ** 2 + 3 * x_plot

plt.plot(x_plot, y_plot, label="f(x) = -x^2 + 3x")
plt.scatter(x_final, -x_final ** 2 + 3 * x_final, color="red", s=50, label="Maximum Point")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Ascent on f(x) = -x^2 + 3x")
plt.show()
