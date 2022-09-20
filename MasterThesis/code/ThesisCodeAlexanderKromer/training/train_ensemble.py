import sys

sys.path.append("..")

from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from loss import LossCalculator, ModelLoss, KnowledgeDistiMode, KnowledgeDistiTeacher
from single_net import get_single_model, initialize_net
from datasets.test_images import get_test_image_loader
from mxnet import gluon
from gluoncv.utils import LRScheduler, LRSequential
import mxnet as mx
from training.util import log_metrics, log_epoch, log_epoch_to_csv, save_current_best_params, save_checkpoint, test, \
    setup_training, finish_batch, finish_epoch, setup_teacher_trainer
from mxnet import autograd, nd
import time
import sys
import logging
from agreement_calculation import Top1Agreement, KLAgreement
from util.util import output_list_to_dict
from attention_matching.attention_matching import init_output_dicts, calculate_attention_loss


def calculate_teacher_output(teacher, x, teacher_ensemble):
    teacher_output = None
    if teacher is not None:
        teacher_output = teacher(x)
        if teacher_ensemble:
            # put outputs in dict, similar to trainee ensemble
            teacher_output = output_list_to_dict(teacher, teacher_output)
    return teacher_output




def train(shared_net, tp, ctx, teacher=None, t_teacher=None, teacher_ensemble=None):
    opt, logger, train_data, val_data, batch_fn, trainer, track_lr, total_time, num_epochs, best_acc, epoch_time, num_examples, summary_writer, teacher_trainer = setup_training(
        tp, ctx, shared_net, teacher)

    t_teacher_trainer = None
    if t_teacher is not None:
        t_teacher_trainer = setup_teacher_trainer(opt, tp, ctx, t_teacher, True)

    # TODO refacto metric creation
    metrics = {}
    for bit_width in shared_net.bit_widths:
        metrics[bit_width] = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval Bit width {bit_width}")

    metrics_test = {}
    for bit_width in shared_net.bit_widths:
        metrics_test[bit_width] = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval Bit width {bit_width}")

    agreement_train_metrics = {}
    agreement_test_metrics = {}
    for bit_width in shared_net.bit_widths:
        agreement_train_metrics[bit_width] = CompositeEvalMetric([Top1Agreement(), KLAgreement()],
                                                                 name=f"Eval Bit width {bit_width}")
        agreement_test_metrics[bit_width] = CompositeEvalMetric([Top1Agreement(), KLAgreement()],
                                                                name=f"Eval Bit width {bit_width}")

    metric_teacher_train = None
    metric_teacher_test = None
    if tp.loss_calculator.knowledge_distillation_teacher_mode != KnowledgeDistiTeacher.none:
        metric_teacher_train = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Teacher train")
        metric_teacher_test = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Teacher test")

    if teacher_ensemble:
        metric_teacher_train = {}
        metric_teacher_test = {}
        for bit_width in shared_net.bit_widths:
            metric_teacher_train[bit_width] = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)],
                                                                  name=f" Teacher Eval Bit width {bit_width}")
            metric_teacher_test[bit_width] = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)],
                                                                 name=f" Teacher Eval Bit width {bit_width}")

    metric_t_teacher_train = None
    metric_t_teacher_test = None
    if t_teacher is not '':
        metric_t_teacher_train = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"T_Teacher train")
        metric_t_teacher_test = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"T_Teacher test")

    warmup_teacher(opt, tp, teacher, teacher_trainer, train_data, val_data, metric_teacher_train, metric_teacher_test, batch_fn, ctx, logger, summary_writer, num_examples, teacher_ensemble)

    for epoch in range(opt.start_epoch, opt.epochs):
        tic = time.time()

        global_step = epoch * num_examples
        if hasattr(train_data, "reset"):
            train_data.reset()
        for metric in metrics.values():
            metric.reset()

        if metric_teacher_train is not None:
            if type(metric_teacher_train) == dict:
                for bit, metric in metric_teacher_train.items():
                    metric.reset()
            else:
                metric_teacher_train.reset()

        if metric_t_teacher_train is not None:
            metric_t_teacher_train.reset()
            metric_t_teacher_test.reset()

        if agreement_train_metrics is not None:
            for metric in agreement_train_metrics.values():
                metric.reset()

        btic = time.time()

        for i, batch in enumerate(train_data):

            if opt.attention_matching != 0:
                init_output_dicts()

            data, label = batch_fn(batch, ctx)
            with autograd.record():
                for x, y in zip(data, label):
                    output = shared_net(x)
                    outputs_dict = output_list_to_dict(shared_net, output)

                    teacher_output = calculate_teacher_output(teacher, x, teacher_ensemble)

                    t_teacher_output = calculate_teacher_output(t_teacher, x, False)

                    # Loss Calculation
                    losses_all = []
                    if tp.loss_calculator.knowledge_distillation_teacher_mode != KnowledgeDistiTeacher.pretrained_external and \
                            tp.loss_calculator.knowledge_distillation_teacher_mode != KnowledgeDistiTeacher.internal:
                        if t_teacher is not None:
                            t_teacher_loss = tp.loss_calculator.calculate_teacher_loss(t_teacher_output, y, False)
                            losses_all.append(t_teacher_loss)

                            teacher_loss = tp.loss_calculator.calulate_ensemble_loss(teacher_output, y,
                                                                                     t_teacher_output, False, True)

                            losses_all.append(teacher_loss)
                        elif teacher is not None:
                            teacher_loss = tp.loss_calculator.calculate_teacher_loss(teacher_output, y,
                                                                                     teacher_ensemble)
                            losses_all.append(teacher_loss)

                    Ls = tp.loss_calculator.calulate_ensemble_loss(outputs_dict, y, teacher_output, teacher_ensemble)
                    losses_all.append(Ls)

                    if opt.attention_matching != 0:
                        att_loss = calculate_attention_loss()
                        losses_all.append(att_loss)

                    autograd.backward(losses_all)

                    # Metric update
                    for bits, output in outputs_dict.items():
                        # TODO HOTFIX when training only 1 bit width and fp weights, refactor
                        if bits not in metrics.keys() and len(shared_net.bit_widths) == 1:
                            metrics[shared_net.bit_widths[0]].update(y, output)
                        else:
                            metrics[bits].update(y, output)

                    if tp.loss_calculator.knowledge_distillation_teacher_mode == KnowledgeDistiTeacher.untrained_external:
                        if type(metric_teacher_train) is dict:
                            for bits, output in teacher_output.items():
                                metric_teacher_train[bits].update(y, output)
                        else:
                            metric_teacher_train.update(y, teacher_output)

                    if teacher_output is not None:
                        for bit in shared_net.bit_widths:
                            if bit in outputs_dict.keys():
                                if teacher_ensemble:
                                    # TODO hotfix when using a single model as ensemble teacher, refactor
                                    if len(teacher_output.keys()) == 1:
                                        t_out = list(teacher_output.values())[0]
                                    else:
                                        t_out = teacher_output[bit]
                                else:
                                    t_out = teacher_output
                                agreement_train_metrics[bit].update(t_out, outputs_dict[bit])

                    if t_teacher_output is not None:
                        metric_t_teacher_train.update(y, t_teacher_output)

            trainer.step(tp.batch_size, ignore_stale_grad=True)

            if t_teacher_trainer is not None:
                t_teacher_trainer.step(tp.batch_size)

            if teacher_trainer is not None:
                teacher_trainer.step(tp.batch_size)

            should_break, global_step, btic = finish_batch(opt, tp, logger, metrics, epoch, summary_writer, global_step,
                                                           i, btic,
                                                           num_examples, tic, epoch_time, track_lr)
            if should_break:
                break

        total_time, num_epochs = finish_epoch(shared_net, tic, summary_writer, ctx, global_step, num_epochs, total_time,
                                              epoch, logger)

        # train
        log_metrics(logger, f"training", metrics, epoch, summary_writer, global_step)

        if tp.loss_calculator.knowledge_distillation_teacher_mode == KnowledgeDistiTeacher.untrained_external:
            if type(metric_teacher_train) is dict:
                log_metrics(logger, f"teacher training", metric_teacher_train, epoch, summary_writer, global_step)
            else:
                log_metrics(logger, f"teacher training", {32: metric_teacher_train}, epoch, summary_writer, global_step)

        if t_teacher is not None:
            log_metrics(logger, f"t_teacher training", {32: metric_t_teacher_train}, epoch, summary_writer, global_step)

        # test
        teacher_output = None
        if tp.loss_calculator.knowledge_distillation_teacher_mode != KnowledgeDistiTeacher.none:
            metric_teacher_test, teacher_output = test(teacher, metric_teacher_test, ctx, val_data, batch_fn,
                                                       opt.test_run, None, True, ensemble_teacher=teacher_ensemble)

        if t_teacher is not None:
            metric_t_teacher_test, _ = test(t_teacher, metric_t_teacher_test, ctx, val_data, batch_fn, opt.test_run,
                                            None, False, None, False)

        acc_target_bits = None
        for bits, metric in metrics_test.items():
            net_b = shared_net.get_net_with_bits(bits)
            # TODO HOTFIX when training only 1 bit width and fp weights, refactor
            if net_b is None and len(shared_net.bit_widths) == 1:
                net_b = shared_net.nets[0]

            test_metric, agreement_test_metrics[bits] = test(net_b, metric, ctx, val_data, batch_fn, opt.test_run,
                                                             teacher_output, False, agreement_test_metrics[bits])
            metrics_test[bits] = test_metric
            if bits == tp.target_bit_width:
                acc_target_bits = test_metric.get()[1][0]

        best_acc = log_epoch(shared_net, logger, metrics, metrics_test, epoch, summary_writer, global_step, tp,
                             logging, trainer, acc_target_bits, best_acc, metric_teacher_train, metric_teacher_test,
                             agreement_train_metrics, agreement_test_metrics, teacher, metric_t_teacher_train,
                             metric_t_teacher_test, t_teacher)

        # post_test_time = time.time()

    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time) / (num_epochs - 1)))


def warmup_teacher(opt, tp, teacher, teacher_trainer, train_data, val_data, metric_teacher_train, metric_teacher_test, batch_fn,
                    ctx, logger, summary_writer, num_examples, teacher_ensemble=False):
    global_step = opt.teacher_warmup_epochs * num_examples
    for epoch in range(opt.start_epoch, opt.teacher_warmup_epochs):
        tic = time.time()

        if hasattr(train_data, "reset"):
            train_data.reset()

        if metric_teacher_train is not None:
            if type(metric_teacher_train) == dict:
                for bit, metric in metric_teacher_train.items():
                    metric.reset()
            else:
                metric_teacher_train.reset()

        for i, batch in enumerate(train_data):

            data, label = batch_fn(batch, ctx)
            with autograd.record():
                for x, y in zip(data, label):
                    teacher_output = calculate_teacher_output(teacher, x, teacher_ensemble)

                    # Loss Calculation

                    teacher_loss = tp.loss_calculator.calculate_teacher_loss(teacher_output, y,
                                                                             teacher_ensemble)
                    autograd.backward(teacher_loss)

                    # Metric update
                    if type(metric_teacher_train) is dict:
                        for bits, output in teacher_output.items():
                            metric_teacher_train[bits].update(y, output)
                    else:
                        metric_teacher_train.update(y, teacher_output)

        teacher_trainer.step(tp.batch_size, ignore_stale_grad=True)
        log_metrics(logger, f"teacher training", {32: metric_teacher_train}, epoch, summary_writer, global_step)

        # test
        teacher_output = None
        metric_teacher_test, teacher_output = test(teacher, metric_teacher_test, ctx, val_data, batch_fn,
                                                       opt.test_run, None, True, ensemble_teacher=teacher_ensemble)

        log_metrics(logger, f"validation", {32: metric_teacher_test}, epoch, summary_writer, global_step)

