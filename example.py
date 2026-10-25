from tensor import Tensor

# Example 1: L = a + b
a = Tensor(5)
b = Tensor(3)
L = a + b

L.backward()
print('Example 1: ', end='')
print(a, b)

# Example 2: L = a * b
a = Tensor(5)
b = Tensor(3)
L = a * b

L.backward()
print('Example 2: ', end='')
print(a, b)

# Example 3: L = (a + b) * a
a = Tensor(5)
b = Tensor(3)
c = a + b
L = c * a

L.grad = 1
L.backward()
print('Example 3: ', end='')
print(a, b)

# Example 4: L = ((a * b) * a) * ((a * b) + b)
a = Tensor(2)
b = Tensor(3)
c = a * b
e = c * a
f = c + b
L = e * f

L.backward()
print('Example 4: ', end='')
print(a, b)

# Example 5
x = Tensor(1)
for _ in range(25):
    x = x * Tensor(1) + x

x.backward()