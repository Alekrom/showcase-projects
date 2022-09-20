import os
import mxnet as mx

from mxnet import autograd
from mxnet.initializer import Constant
from mxnet.autograd import Function
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import base
from mxnet import nd


class PACT(HybridBlock):
    def __init__(self, bits=1, cgpact=False, **kwargs):
        super(PACT, self).__init__(**kwargs)
        with self.name_scope():
            self.bits = bits
            self.cgpact = cgpact
            # https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py line 113 inits alpha to bits * 8.0
            initializer = Constant(8.0)
            self.alpha = self.params.get('alpha', shape=(1,), init=initializer, allow_deferred_init=True)

    def _q(self, x):
        max_v = (2 ** self.bits) - 1
        return nd.round(x * max_v) / max_v

    def hybrid_forward(self, F, x, alpha):
        with autograd.pause():
            x_clipped = nd.clip(x, 0, alpha.asscalar())
            q = nd.broadcast_mul(alpha, self._q(nd.broadcast_div(x_clipped, alpha)))

        # fix gradient of alpha
        ste = nd.greater(x, alpha)
        if self.cgpact:
            # replaces zero returned for x < alpha in straight through estimator
            x_clipped = nd.clip(x.detach(), 0, alpha.asscalar())
            x_clipped_div_alpha = nd.broadcast_div(x_clipped, alpha.detach())
            quan = self._q(x_clipped_div_alpha)
            v = nd.broadcast_sub(quan, x_clipped_div_alpha)
            # replace zeros with v (0 in ste -> cond is False)
            ste = mx.nd.where(ste, ste, v)
        ste = ste * alpha

        # fix gradient of x
        x_grad_fn = nd.clip(x, 0, alpha.asscalar())

        out = (q - ste - x_grad_fn).detach() + ste + x_grad_fn

        return out


class PactActivationBuilder:
    def __init__(self, bits, cgpact):
        self.bits = bits
        self.cgpact = cgpact

    def get_pact(self):
        return PACT(self.bits, self.cgpact)
