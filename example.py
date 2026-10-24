from tensor import Tensor

a = Tensor(5)
b = Tensor(3)
c = a * b
L = a * a

L.grad = 1  # L = a * b * a
# c.grad = L.grad * Tensor(3).data  # dL/dL * dL/dc = 3
# a.grad = c.grad * b.data  # dL/dL * dL/dc * dc/da
# b.grad = c.grad * a.data  # dL/dL * dL/dc * dc/db

L._backward()
print(a)