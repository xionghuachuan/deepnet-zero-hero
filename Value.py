import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import math

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


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
