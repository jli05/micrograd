
from numpy import (array, ndarray, nan, ones, zeros, full,
                   shape as get_shape, where, sum as npsum,
                   log1p, arctanh, broadcast_arrays, expand_dims,
                   prod, tensordot, isnan, all as npall)
from numbers import Number
from warnings import warn

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data=None, _children=(), _op='',
                 shape=None, name=None):
        if data is not None:
            assert isinstance(data, (ndarray, Number))
            assert name is None, "data provided, no need for name"
            assert shape is None, "data provided, no need for shape"
            self.data = data
            self.name = None
            self.shape = get_shape(data)
        else:
            assert name, "data not provided, name must be given"
            assert shape is not None, "data not provided, shape must be given"
            self.name = name
            self.shape = shape
            self.data = full(self.shape, nan)
        self.grad = None
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

        def _forward(**kwds):
            if self.name:
                if self.name in kwds:
                    _value = kwds[self.name]
                    assert isinstance(_value, (ndarray, Number))
                    assert get_shape(_value) == self.shape
                    self.data = _value
                else:
                    warn(f'{self.name} not in input data')
                    self.data = full(self.shape, nan)
        self._forward = _forward

    def __add__(self, other):
        other = (other if isinstance(other, Value)
                 else Value(other, _op='c'))
        out = Value(self.data + other.data, (self, other), '+')

        def _forward(**kwds):
            out.data = self.data + other.data
        out._forward = _forward

        def _backward():
            if self.shape == ():
                self.grad += npsum(out.grad)
            else:
                self.grad += out.grad
            if other.shape == ():
                other.grad += npsum(out.grad)
            else:
                other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = (other if isinstance(other, Value)
                 else Value(other, _op='c'))
        out = Value(self.data * other.data, (self, other), '*')

        def _forward(**kwds):
            out.data = self.data * other.data
        out._forward = _forward

        def _backward():
            if self.shape == ():
                self.grad += npsum(other.data * out.grad)
            else:
                self.grad += other.data * out.grad
            if other.shape == ():
                other.grad += npsum(self.data * out.grad)
            else:
                other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # TOODO: array(3) ** -1 won't do. array(3).astype(float) if excepted
        assert isinstance(other, (int, float)), ("only supporting"
                                                 " int/float powers for now")
        out = Value(self.data ** other, (self,), f'**{other}')

        def _forward(**kwds):
            out.data = self.data ** other
        out._forward = _forward

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    @property
    def T(self):
        out = Value(self.data.T, (self,), 'T')

        def _forward(**kwds):
            out.data = self.data.T
        out._forward = _forward

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out

    @property
    def ndim(self):
        return len(self.shape)

    def relu(self):
        out = Value(where(self.data > 0, self.data, 0), (self,), 'ReLU')

        def _forward(**kwds):
            out.data = where(self.data > 0, self.data, 0)
        out._forward = _forward

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def log1p(self):
        out = Value(log1p(self.data), (self,), 'log1p')

        def _forward(**kwds):
            out.data = log1p(self.data)
        out._forward = _forward

        def _backward():
            valid_data = where(self.data >= 0, self.data, nan)
            self.grad += 1 / (1 + valid_data) * out.grad
        out._backward = _backward

        return out

    def arctanh(self):
        out = Value(arctanh(self.data), (self,), 'arctanh')

        def _forward(**kwds):
            out.data = arctanh(self.data)
        out._forward = _forward

        def _backward():
            arctanh_grad = 1 / (1 - self.data ** 2)
            arctanh_grad = where(arctanh_grad >= 1, arctanh_grad, nan)
            self.grad += arctanh_grad * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None):
        out = Value(npsum(self.data, axis=axis), (self,), 'sum')

        if axis is None:
            new_shape = self.shape
        elif isinstance(axis, int):
            new_shape = [h if j == axis else 1
                         for j, h in enumerate(self.shape)]
        else:
            new_shape = [h if j in axis else 1
                         for j, h in enumerate(self.shape)]
        m_new = ones(new_shape)

        expand_axis = tuple(range(self.data.ndim)) if axis is None else axis

        def _forward(**kwds):
            out.data = npsum(self.data, axis=axis)
        out._forward = _forward

        def _backward():
            self.grad += broadcast_arrays(expand_dims(out.grad, expand_axis),
                                          m_new)[0]
        out._backward = _backward

        return out

    def mean(self, axis=None):
        shape_arr = array(self.shape)
        if axis is None:
            denom = prod(shape_arr)
        elif isinstance(axis, int):
            denom = shape_arr[axis]
        else:
            denom = prod(shape_arr[list(axis)])

        return self.sum(axis) * (1 / denom)

    def tensordot(self, other, axes):
        ''' Tensor contraction, only accepting int axes '''
        assert axes >= 0          # only int axes
        axes1 = [[-1 - j for j in range(axes)], list(range(axes))]
        axes2 = [[-1 - j for j in range(self.ndim - axes)],
                 list(range(self.ndim - axes))]
        axes3 = [[-1 - j for j in range(other.ndim - axes)],
                 list(range(other.ndim - axes))]

        other = (other if isinstance(other, Value)
                 else Value(other, _op='c'))
        out = Value(tensordot(self.data, other.data, axes=axes1),
                    (self, other), '@')

        def _forward(**kwds):
            out.data = tensordot(self.data, other.data, axes=axes1)
        out._forward = _forward

        def _backward():
            self.grad += tensordot(out.grad, other.data.T, axes=axes3)
            other.grad += tensordot(self.data.T, out.grad, axes=axes2)
        out._backward = _backward

        return out

    def build_topology(self):
        # topological order all of the children in the graph
        if not hasattr(self, 'topo'):
            self.topo = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    self.topo.append(v)

            build_topo(self)

    def forward(self, **kwds):

        self.build_topology()
        for v in self.topo:
            v._forward(**kwds)

    def backward(self):

        if npall(isnan(self.data)):
            warn('run forward() before backward()')

        self.build_topology()
        # go one variable at a time and apply the chain rule to get its gradient
        for v in self.topo:
            v.grad = ones(self.shape) if v == self else zeros(v.shape)
        for v in reversed(self.topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __matmul__(self, other):
        return self.tensordot(other, 1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
