from tensor import Tensor

def f(x):
    return x**2 - 1

x = Tensor(2)
for _ in range(1000):
    y = f(x)
    x.grad = 0
    y.backward()
    x.data -= x.grad * 0.01
    print(x)