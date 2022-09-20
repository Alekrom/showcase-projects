"""

    Not Maintained, consider removing

"""


import sys
sys.path.append("..")

from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from loss import LossCalculator, ModelLoss
from single_net import get_single_model, initialize_net
from datasets.test_images import get_test_image_loader
from mxnet import gluon
from datasets.util import *
from gluoncv.utils import LRScheduler, LRSequential
import mxnet as mx
from training.util import get_optimizer, LRTracker, write_net_summaries, log_metrics, log_epoch_to_csv, save_current_best_params, save_checkpoint, log_epoch, test, setup_training, finish_batch, finish_epoch
from mxnet import autograd
import time
import logging


def train(net, tp, ctx):
    opt, logger, train_data, val_data, batch_fn, trainer, track_lr, total_time, num_epochs, best_acc, epoch_time, num_examples, summary_writer, teacher_trainer = setup_training(
        tp, ctx, net)

    bits = net.bits
    metrics = {}
    metrics[net.bits] = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval Bit width {bits}")

    metrics_test = {}
    metrics_test[bits] = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval Bit width {bits}")

    print("start training single")
    for epoch in range(opt.start_epoch, opt.epochs):
        print("currently broken, agreement")
        tic = time.time()

        global_step = epoch * num_examples
        if hasattr(train_data, "reset"):
            train_data.reset()
        for metric in metrics.values():
            metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)
            with autograd.record():
                Ls = []
                for x, y in zip(data, label):
                    # This is only executed once
                    z = net(x)
                    L = tp.loss_fn(z, y)
                    Ls.append(L)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                autograd.backward(Ls)
            trainer.step(tp.batch_size)

            # Update metrics
            metrics[bits].update(y, z)

            should_break, global_step, btic = finish_batch(opt, tp, logger, metrics, epoch, summary_writer, global_step, i, btic, num_examples, tic, epoch_time, track_lr)
            if should_break:
                break

        total_time, num_epochs = finish_epoch(net, tic, summary_writer, ctx, global_step, num_epochs, total_time, epoch, logger)

        # train
        log_metrics(logger, f"training", metrics, epoch, summary_writer, global_step)

        # test
        metrics_test[bits], _ = test(net, metrics_test[bits], ctx, val_data, batch_fn, opt.test_run)
        current_acc = metrics_test[bits].get()[1][0]

        best_acc = log_epoch(net, logger, metrics, metrics_test, epoch, summary_writer, global_step, tp, logging, trainer, current_acc, best_acc)

    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))