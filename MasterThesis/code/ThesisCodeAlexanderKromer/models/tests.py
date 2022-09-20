import os

import logging
import sys

# If the python binding doesnt work mxnet can be added manually
#sys.path.insert(1, '/work/mxnet/python/')

from mxnet import nd
from mxnet import autograd
from mxnet.gluon import loss as gloss
from mxnet.test_utils import assert_almost_equal, check_numeric_gradient, numeric_grad
import numpy as np
from mxnet.gluon.loss import KLDivLoss
from mxnet import gluon
import mxnet

from pact import PACT
"""
Tests for some of the new components
"""

def test_pact_simple_binary():
    # initial alpha is 8.0
    cgpact = False
    bits = 1

    exp_grad_x = np.array([1.0, 0.0, 0.0])
    exp_grad_alpha = np.array([1.0])
    exp_y = np.array([8.0, 8.0, 0])

    pact_l = PACT(bits, cgpact)
    pact_l.initialize()

    x = nd.array([7.2, 19.0, -2.3])
    x.attach_grad()
    with autograd.record():
        y = pact_l(x)
    y.backward()

    assert_almost_equal(y.asnumpy(), exp_y)
    assert_almost_equal(x.grad.asnumpy(), exp_grad_x)
    assert_almost_equal(pact_l.alpha.grad().asnumpy(), exp_grad_alpha)


def test_pact_simple_32_bit():
    # initial alpha is 8.0
    cgpact = False
    bits = 32

    exp_grad_x = np.array([1.0, 0.0, 0.0])
    exp_grad_alpha = np.array([1.0])
    exp_y = np.array([7.2, 8.0, 0])

    pact_l = PACT(bits, cgpact)
    pact_l.initialize()

    x = nd.array([7.2, 19.0, -2.3])
    x.attach_grad()
    with autograd.record():
        y = pact_l(x)
    y.backward()

    assert_almost_equal(y.asnumpy(), exp_y)
    assert_almost_equal(x.grad.asnumpy(), exp_grad_x)
    assert_almost_equal(pact_l.alpha.grad().asnumpy(), exp_grad_alpha)


def test_pact_simple_32_bit_modified_loss():
    # initial alpha is 8.0
    cgpact = False
    bits = 32

    exp_grad_x = np.array([1/3, 0.0, 0.0])
    exp_grad_alpha = np.array([1/3])
    exp_y = nd.array([7.2, 8.0, 0])
    exp_y = nd.expand_dims(exp_y, axis=0)

    pact_l = PACT(bits, cgpact)
    pact_l.initialize()

    x = nd.array([7.2, 19.0, -2.3])
    x = nd.expand_dims(x, axis=0)
    x.attach_grad()
    with autograd.record():
        y = pact_l(x)

        loss = gloss.L1Loss()
        #label = nd.array([0, 0, 0])
        label = nd.zeros((1,3))
        l = loss(y, label)
        autograd.backward(l)

    assert_almost_equal(y.asnumpy(), exp_y.asnumpy())
    assert_almost_equal(x.grad.asnumpy(), exp_grad_x)
    assert_almost_equal(pact_l.alpha.grad().asnumpy(), exp_grad_alpha)


def test_pact_gc_binary():
    # initial alpha is 8.0
    cgpact = True
    bits = 1

    exp_grad_x = np.array([1.0, 0.0, 0.0])
    exp_grad_alpha = np.array([1.1])
    exp_y = np.array([8.0, 8.0, 0])

    pact_l = PACT(bits, cgpact)
    pact_l.initialize()

    x = nd.array([7.2, 19.0, -2.3])
    x.attach_grad()
    with autograd.record():
        y = pact_l(x)
    y.backward()

    assert_almost_equal(y.asnumpy(), exp_y)
    assert_almost_equal(x.grad.asnumpy(), exp_grad_x)
    assert_almost_equal(pact_l.alpha.grad().asnumpy(), exp_grad_alpha)


def test_pact_gc_32_binary():
    # initial alpha is 8.0
    cgpact = True
    bits = 32

    exp_grad_x = np.array([1.0, 0.0, 0.0])
    exp_grad_alpha = np.array([1.0])
    exp_y = np.array([7.2, 8.0, 0])

    pact_l = PACT(bits, cgpact)
    pact_l.initialize()

    x = nd.array([7.2, 19.0, -2.3])
    x.attach_grad()
    with autograd.record():
        y = pact_l(x)
    y.backward()

    assert_almost_equal(y.asnumpy(), exp_y)
    assert_almost_equal(x.grad.asnumpy(), exp_grad_x)
    assert_almost_equal(pact_l.alpha.grad().asnumpy(), exp_grad_alpha)


def test_gradient_calc():
    x1 = nd.array([[1.0], [1.0], [1.0]])
    x2 = nd.array([[0.0], [0.0], [0.0]])
    x1.attach_grad()
    x2.attach_grad()
    label = nd.array([1.0, 1.0, 1.0])
    with autograd.record():
        loss = gloss.L2Loss()

        l1 = loss(x1, label)
        l1 = nd.expand_dims(l1, axis=0)
        l2 = loss(x2, label)
        l2 = nd.expand_dims(l2, axis=0)
        l_all = nd.concat(l1, l2, dim=0)
    autograd.backward(l_all)
    print("ok")


def test_kl_div_loss():
    loss_fn = KLDivLoss(from_logits=False)
    fp_loss = nd.array([1.964879, 2.2290921, 2.616047, 3.1204197]).expand_dims(0)
    fp_loss = nd.softmax(fp_loss)
    loss_same = nd.array([1.964879, 2.2290921, 2.616047, 3.1204197]).expand_dims(0)
    #loss_same = loss_same.log()
    loss_diff = nd.array([2.0303156, 2.1526601, 2.9101002, 3.482172]).expand_dims(0)
    #loss_diff = loss_diff.log()
    y1 = loss_fn(loss_same, fp_loss)
    y2 = loss_fn(loss_diff, fp_loss)
    y3 = loss_fn(fp_loss, fp_loss)
    print("ok")


def test_weight_decay():
    # initial alpha is 8.0
    cgpact = True
    bits = 32

    exp_alpha = 0

    pact_l = PACT(bits, cgpact)
    pact_l.initialize()

    pact_l.alpha.wd_mult = 2.0

    x = nd.array([7.2, 7.2, 7.2])

    optimizer = mxnet.optimizer.Adam(learning_rate=0.1, wd=0.0)
    params = pact_l.collect_params()
    trainer = gluon.Trainer(pact_l.collect_params(), optimizer)

    x.attach_grad()
    with autograd.record():
        y = pact_l(x)
    y.backward()
    trainer.step(1)
    aux = pact_l.alpha.data()
    aux2 = pact_l.alpha.grad().asnumpy()
    assert_almost_equal(pact_l.alpha.data(), exp_alpha)

    with autograd.record():
        y = pact_l(x)
    y.backward()



if __name__ == '__main__':
    #test_gradient_calc()
    #test_pact_simple_binary()
    #test_pact_simple_32_bit()
    #test_pact_gc_binary()
    #test_pact_gc_32_binary()
    #test_pact_simple_32_bit_modified_loss()
    test_kl_div_loss()
    #test_weight_decay()