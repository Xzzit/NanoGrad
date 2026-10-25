class Tensor:
    def __init__(self, data, _pre=()) -> None:
        self.data = data
        self.grad = 0

        self._pre = set(_pre)
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(data: {self.data}, grad: {self.grad})"

    @staticmethod
    def to_tensor(x):
        if isinstance(x, Tensor):
            return x
        else:
            return Tensor(x)

    # Case: x + y
    def __add__(self, other):
        other = self.to_tensor(other)

        data = self.data + other.data
        out = Tensor(data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self.__add__(other)

    # Case: x - y
    def __sub__(self, other):
        other = self.to_tensor(other)

        data = self.data - other.data
        out = Tensor(data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward

        return out

    def __rsub__(self, other):
        other = self.to_tensor(other)

        data = other.data - self.data
        out = Tensor(data, (self, other))

        def _backward():
            self.grad -= out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __neg__(self):

        data = -self.data
        out = Tensor(data, (self, ))

        def _backward():
            self.grad -= out.grad
        out._backward = _backward

        return out

    # Case: x * y
    def __mul__(self, other):
        other = self.to_tensor(other)

        data = self.data * other.data
        out = Tensor(data, (self, other))

        def _backward():
            self.grad += (other.data * out.grad)
            other.grad += (self.data * out.grad)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    # Case: x ** n, where n is a constant (int or float)
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Only constant (int or float) exponents are supported.")

        data = self.data ** other
        out = Tensor(data, (self, ))

        def _backward():
            self.grad += (other * (self.data ** (other - 1)) * out.grad)
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node in visited:
                return
            else:
                visited.add(node)
            
            for n in node._pre:
                build(n)

            topo.append(node)

        self.grad = 1.0
        build(self)

        for node in reversed(topo):
            node._backward()