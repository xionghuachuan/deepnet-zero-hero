#构建反向传播，需要构建计算图，每一个变量就是一个node
#

class Value:
    def __init__(self, data, op= '', label=''):
        self.data = data
        self.grad = 0
        self.label = label
        self._op = op
        self._prev = set()
        self._backward = lambda :None

    def __repr__(self):
        return f'Value({self.label}: {self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data)
        out._prev = {self, other}
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        out._op = '+'
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data)
        out._prev = {self, other}
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        out._op = '*'
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data - other.data)
        out._prev = {self, other}
        def backward():
            self.grad += out.grad
            other.grad += - 1.0 * out.grad

        out._backward = backward
        out._op = '-'
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data / other.data)
        out._prev = {self, other}
        def backward():
            self.grad += (other.data ** (-1.0)) * out.grad
            other.grad += - 1.0 * (other.data ** (-2.0)) * out.grad

        out._backward = backward
        out._op = '/'
        return out

    def tanh(self):
        ex = math.exp(self.data)
        enx = math.exp(-self.data)

        out = Value((ex - enx) / (ex + enx))
        out._prev = {self}
        def backward():
            self.grad += (1- self.tanh().data ** 2) * out.grad
        out._backward = backward
        out._op = 'tanh'
        return out

    def backward(self):
        self._backward()
        for node in self._prev:
            assert isinstance(node, Value)
            node.backward()
