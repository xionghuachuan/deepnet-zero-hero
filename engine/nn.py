## 模型更新一次梯度之后需要清除梯度，否则梯度会累加

from engine.Value import Value
import numpy as np
from abc import ABC, abstractmethod


class Module:
    def __init__(self):
        self._parameters = set()

    @abstractmethod
    def parameters(self):
        pass

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0


class Neuron():
    def __init__(self, nin, nonlin=True):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self._parameters = set(self.w + [self.b])
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh() if self.nonlin else act
        return out

    def parameters(self):
        return self._parameters


class Layer():
    def __init__(self, nin, nout, **kwargs):
        self._neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        self._parameters = set()

    def __call__(self, x):
        out = [n(x) for n in self._neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        for neuron in self._neurons:
            self._parameters.update(neuron.parameters())  ### set update 为添加元素，参数可以是任意iterable对象
        return self._parameters


class MLP(Module):

    def __init__(self, nin, nouts):
        super().__init__()
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
