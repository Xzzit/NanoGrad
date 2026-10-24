class Tensor:
    def __init__(self, data, _pre=()) -> None:
        self.data = data
        self.grad = 0

        self._pre = tuple(_pre)
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(data: {self.data}, grad: {self.grad})"

    def __add__(self, other):
        data = self.data + other.data
        return Tensor(data, (self, other))

    def __mul__(self, other):
        data = self.data * other.data
        out = Tensor(data, (self, other))

        def _backward():
            self.grad += (other.data * out.grad)
            other.grad += (self.data * out.grad)
        out._backward = _backward

        return out