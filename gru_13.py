import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((x.shape[0], x.shape[1] // 6, 6) + x.shape[2:])
    return (r[:, :, i] for i in six.moves.range(6))


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class GRU(function.Function):

    """Long short-term memory unit with forget gate.

    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    """
    def forward(self, inputs):
        h,c_prev, x = inputs
        #a, i, f, o = _extract_gates(x)
        W_r,U_r,W_z,U_z,W,U = _extract_gates(x)

        if isinstance(x, numpy.ndarray):
            #self.a = numpy.tanh(a)
            #self.i = _sigmoid(i)
            #self.f = _sigmoid(f)
            #self.o = _sigmoid(o)
            r = _sigmoid(self.W_r(c_prev) + self.U_r(h))
            z = _sigmoid(self.W_z(c_prev) + self.U_z(h))
            h_bar = tanh.tanh(self.W(c_prev) + self.U(r * h))
            h_new = (1 - z) * h + z * h_bar
            self.c = h_new

            #self.c = self.a * self.i + self.f * c_prev
            x_new = c_prev * numpy.tanh(self.c) 
        else:
            self.c, h = cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o', 'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af * c_prev;
                    h = ao * tanh(c);
                ''',
                'lstm_fwd', preamble=_preamble)(h,c_prev,W_r,U_r,W_z,U_z,W,U)

        return self.c,x_new

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x = inputs
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            gc_prev = gh * self.o * _grad_tanh(co) + gc  # multiply f later
            ga[:] = gc_prev * self.i * _grad_tanh(self.a)
            gi[:] = gc_prev * self.a * _grad_sigmoid(self.i)
            gf[:] = gc_prev * c_prev * _grad_sigmoid(self.f)
            go[:] = gh * co * _grad_sigmoid(self.o)
            gc_prev *= self.f  # multiply f here
        else:
            a, i, f, o = _extract_gates(x)
            gc_prev = xp.empty_like(c_prev)
            cuda.elementwise(
                'T c_prev, T c, T gc, T gh, T a, T i_, T f, T o',
                'T gc_prev, T ga, T gi, T gf, T go',
                '''
                    COMMON_ROUTINE;
                    T co = tanh(c);
                    T temp = gh * ao * grad_tanh(co) + gc;
                    ga = temp * ai * grad_tanh(aa);
                    gi = temp * aa * grad_sigmoid(ai);
                    gf = temp * c_prev * grad_sigmoid(af);
                    go = gh * co * grad_sigmoid(ao);
                    gc_prev = temp * af;
                ''',
                'lstm_bwd', preamble=_preamble)(
                    c_prev, self.c, gc, gh, a, i, f, o,
                    gc_prev, ga, gi, gf, go)

        return gc_prev, gx


def lstm(c_prev, x):
    return LSTM()(c_prev, x)