import copy
import os
import weakref

import numpy
import six

from chainer import cuda
from chainer.utils import type_check
from chainer import variable


class Function2(object):
    parameter_names = ()
    gradient_names = ()
    type_check_enable = int(os.environ.get('CHAINER_TYPE_CHECK', '1')) != 0
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.rank = None
    def __call__(self, *inputs):
        # First copy itself to avoid duplication within the graph.
        self = copy.copy(self)
        inputs = inputs[0]
        if any(x.volatile for x in inputs):  # not build graph
            # do not mix multiple volatility
            assert all(x.volatile for x in inputs)
            in_data = tuple(x.data for x in inputs)
            if self.type_check_enable:
                self._check_data_type_forward(in_data)
            with cuda.get_device(*in_data):
                out_data = self.forward(in_data)
            assert type(out_data) == tuple
            outputs = list(variable.Variable(y, volatile=True)
                           for y in out_data)
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        # Build graph
        # Be careful that forward references must be weak
        self.inputs = []
        for x in inputs:
            splitter = x.splitter()
            if splitter is None:
                splitter = Split(x)
                x.splitter = weakref.ref(splitter)
            self.inputs.append(splitter.add_branch())
        if self.inputs:
            self.rank = max(x.rank for x in self.inputs)
        else:
            self.rank = 0
        in_data = tuple(x.data for x in self.inputs)
        if self.type_check_enable:
            self._check_data_type_forward(in_data)
        with cuda.get_device(*in_data):
            outputs = self.forward(in_data)
        assert type(outputs) == tuple
        ret = tuple(variable.Variable(y) for y in outputs)
        for y in ret:
            y.set_creator(self)
        # Make forward references weak
        self.outputs = tuple(weakref.ref(y) for y in ret)
        if len(ret) == 1:
            return ret[0]
        return ret
    @property
    def label(self):
        return self.__class__.__name__
    def _check_data_type_forward(self, in_data):
        in_type = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_type)
    def check_type_forward(self, in_types):
        """Checks types of input data before forward propagation.
        Before :meth:`forward` is called, this function is called.
        You need to validate types of input data in this function
        using :ref:`the type checking utilities <type-check-utils>`.
        Args:
            in_types (~chainer.utils.type_check.TypeInfoTuple): The type
                information of input data for :meth:`forward`.
        """
        pass
    def forward(self, inputs):
        if any(isinstance(x, cuda.ndarray) for x in inputs):
            return self.forward_gpu(inputs)
        else:
            return self.forward_cpu(inputs)
    def forward_cpu(self, inputs):
        raise NotImplementedError()
    def forward_gpu(self, inputs):
        raise NotImplementedError()
    def backward(self, inputs, grad_outputs):
        if any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs):
            return self.backward_gpu(inputs, grad_outputs)
        else:
            return self.backward_cpu(inputs, grad_outputs)
    def backward_cpu(self, inputs, grad_outputs):
        return tuple(None for _ in inputs)
    def backward_gpu(self, inputs, grad_outputs):
        return tuple(None for _ in inputs)
    def unchain(self):
        for y in self.outputs:
            y_ref = y()
            if y_ref is not None:
                y_ref.creator = None
        for x in self.inputs:
            x.splitter = weakref.ref(lambda: 0)  # dead ref
        self.inputs = None
    def to_gpu(self, device=None):
        with cuda.get_device(device):
            for k, v in six.iteritems(self.__dict__):
                if isinstance(v, numpy.ndarray):
                    setattr(self, k, cuda.cupy.array(v))
        return self
    def to_cpu(self):
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, cuda.ndarray):
                setattr(self, k, v.get())
        return self
    @property
    def parameters(self):
        return tuple(getattr(self, name) for name in self.parameter_names)
    @parameters.setter
    def parameters(self, values):
        assert len(self.parameter_names) == len(values)
        for name, value in zip(self.parameter_names, values):
            setattr(self, name, value)
    @property
    def gradients(self):
        return tuple(getattr(self, name) for name in self.gradient_names)
    @gradients.setter
    def gradients(self, values):
        assert len(self.gradient_names) == len(values)
        for name, value in zip(self.gradient_names, values):
            setattr(self, name, value)


class Split(Function2):
    def __init__(self, var):
        self.inputs = [var]
        self.outputs = []
        self.rank = var.rank
    def add_branch(self):
        x = self.inputs[0]
        output = variable.Variable(x.data)
        output.set_creator(self)
        self.outputs.append(weakref.ref(output))
        return output
    def backward(self, inputs, grad_outputs):
        # Accumulate gradients
        if len(grad_outputs) == 1:
            return grad_outputs  # no copy
        gx = None
        grad_outputs = [gy for gy in grad_outputs if gy is not None]
        with cuda.get_device(*grad_outputs):
            for gy in grad_outputs:
                if gx is None:
                    gx = gy.copy()
                else:
                    gx += gy
        return gx,
