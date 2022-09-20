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
from mxnet.test_utils import assert_almost_equal
import math

from agreement_calculation import Top1Agreement, KLAgreement

def test_top1_agreement():
    teacher_pred = nd.array([[1.0, 2.0, -1.0],
                            [-1.0, 2.0, -1.0],
                            [1.0, -2.0, 2.0],
                             [1.0, -2.0, 2.0]])

    student_pred = nd.array([[1.0, -2.0, -1.0],
                            [1.0, 2.0, 1.0],
                            [1.0, -1.0, 2.0],
                             [1.0, -2.0, 2.0]])

    metric = Top1Agreement()
    metric.update(teacher_pred, student_pred)
    agreement = metric.get()[1]
    assert agreement == 0.75

    teacher_pred = nd.array([[1.0, 2.0, -1.0],
                            [-1.0, 2.0, -1.0],
                            [1.0, -2.0, 2.0],
                             [1.0, -2.0, 2.0]])

    student_pred = nd.array([[1.0, -2.0, -1.0],
                            [1.0, -2.0, 1.0],
                            [1.0, -1.0, -2.0],
                             [1.0, -2.0, 2.0]])
    metric.update(teacher_pred, student_pred)
    agreement = metric.get()[1]
    assert agreement == 0.5

    metric.reset()
    agreement = metric.get()[1]
    assert math.isnan(agreement)

def test_kl_div_agreement():
    teacher_pred = nd.array([[1.0, 2.0, -1.0],
                             [-1.0, 2.0, -1.0],
                             [1.0, -2.0, 2.0],
                             [1.0, -2.0, 2.0]])

    teacher_pred = nd.array([-1.0, 2.0, -1.0])

    student_pred = nd.array([[1.0, -2.0, -1.0],
                             [1.0, 2.0, 1.0],
                             [1.0, -1.0, 2.0],
                             [1.0, -2.0, 2.0]])

    metric = KLAgreement()
    metric.update(teacher_pred, teacher_pred)
    agreement = metric.get()[1]
    assert_almost_equal(nd.array([agreement]), nd.array([0.0]))

    teacher_pred = nd.array([[1.0, 2.0, -1.0],
                             [-1.0, 2.0, -1.0],
                             [1.0, -2.0, 2.0],
                             [1.0, -2.0, 2.0]])

    student_pred = nd.array([[1.0, -2.0, -1.0],
                             [1.0, -2.0, 1.0],
                             [1.0, -1.0, -2.0],
                             [1.0, -2.0, 2.0]])
    metric.update(teacher_pred, student_pred)
    agreement = metric.get()[1]
    assert_almost_equal(agreement, 0.24002046883106232)

    metric.reset()
    agreement = metric.get()[1]
    assert math.isnan(agreement)


if __name__ == '__main__':
    test_top1_agreement()
    test_kl_div_agreement()