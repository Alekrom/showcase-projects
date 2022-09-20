import sys
sys.path.append("..")

from datasets.util import *
from gluoncv.utils import LRScheduler, LRSequential
import csv
from datetime import datetime
import os
import sys
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
import mxnet as mx
from mxnet import gluon
from mxnet import nd
import time
from loss import KnowledgeDistiMode, KnowledgeDistiTeacher
from util.log_progress import log_progress
from mxnet import autograd


class TrainingParams():
    def __init__(self, opt, batch_size, logger, loss_fn, loss_calculator, fh_csv, best_checkpoint_path, mxboard_summary, target_bit_width, no_batch_reuse):
        self.opt = opt
        self.batch_size = batch_size
        self.mxboard_summary = mxboard_summary
        self.logger = logger
        self.loss_fn = loss_fn
        self.loss_calculator = loss_calculator
        self.fh_csv = fh_csv
        self.best_checkpoint_path = best_checkpoint_path
        self.target_bit_width = target_bit_width
        self.no_batch_reuse = no_batch_reuse

class LRScheduleParams():
    def __init__(self, lr_mode, lr, lr_factor, lr_steps, warmup_epochs, epochs, dataset):
        self.lr_mode = lr_mode
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_steps = lr_steps
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.dataset = dataset

class OptimizerParams():
    def __init__(self, optimizer, wd, momentum, dtype='float32', activation_method='round', wd_alpha=None):
        self.optimizer = optimizer
        self.wd = wd
        self.momentum = momentum
        self.dtype = dtype
        self.activation_method = activation_method
        self.wd_alpha = wd_alpha


def _get_lr_scheduler(lr_params, batch_size):
    lr_factor = lr_params.lr_factor
    lr_steps = [int(i) for i in lr_params.lr_steps.split(',')]
    lr_steps = [e - lr_params.warmup_epochs for e in lr_steps]
    num_batches = get_num_examples(lr_params.dataset) // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=lr_params.lr,
                    nepochs=lr_params.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(lr_params.lr_mode, base_lr=lr_params.lr, target_lr=0,
                    nepochs=lr_params.epochs - lr_params.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_steps,
                    step_factor=lr_factor, power=2)
    ])
    return lr_scheduler

def get_optimizer(optimizer_params, lr_params, batch_size):
    if optimizer_params.activation_method == 'pact':
        params = {
            'wd': optimizer_params.wd_alpha,
            'lr_scheduler': _get_lr_scheduler(lr_params, batch_size)
        }
    else:
        params = {
            'wd': optimizer_params.wd,
            'lr_scheduler': _get_lr_scheduler(lr_params, batch_size)
        }

    if optimizer_params.dtype != 'float32':
        params['multi_precision'] = True
    if optimizer_params.optimizer == "sgd" or optimizer_params.optimizer == "nag":
        params['momentum'] = optimizer_params.momentum
    return optimizer_params.optimizer, params


class LRTracker:
    def __init__(self, trainer, summary_writer, logger):
        self.trainer = trainer
        self.prev_lr = trainer.learning_rate
        self.summary_writer = summary_writer
        self.logger = logger

    def __call__(self, epoch, global_step=0):
        current_lr = self.trainer.learning_rate
        if current_lr != self.prev_lr:
            self.logger.info('[Epoch %d] Change learning rate to %f', epoch, current_lr)
            self.prev_lr = current_lr
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("training/lr", current_lr, global_step=global_step)


def write_net_summaries(net, summary_writer, single_ctx, global_step=0, write_grads=True):
    if summary_writer is None:
        return

    params = net.collect_params(".*weight|.*bias|.*alpha")
    for name, param in params.items():
        summary_writer.add_histogram(tag=name, values=param.data(single_ctx),
                                     global_step=global_step, bins=1000)
        if write_grads:
            summary_writer.add_histogram(tag="%s-grad" % name, values=param.grad(single_ctx),
                                         global_step=global_step, bins=1000)


def log_epoch_to_csv(metrics_train, metrics_test, epoch, fh_csv, metric_teacher_train=None, metric_teacher_test=None,
                     agreement_train_metrics=None, agreement_test_metrics=None,  metric_t_teacher_train=None, metric_t_teacher_test=None):
    for bits in metrics_train.keys():
        name_train, acc_train = metrics_train[bits].get()
        name_test, acc_test = metrics_test[bits].get()
        with open(fh_csv, mode='a') as results_csv:
            r_csv_writer = csv.writer(results_csv, delimiter=',')
            row = [epoch, bits, acc_train[0], acc_train[1], acc_test[0], acc_test[1]]
            if agreement_train_metrics is not None and agreement_test_metrics is not None:
                name_agr_train, agr_train = agreement_train_metrics[bits].get()
                name_agr_test, agr_test = agreement_test_metrics[bits].get()
            row = row + [agr_train[0], agr_train[1], agr_test[0], agr_test[1]]
            r_csv_writer.writerow(row)

    if metric_teacher_test is not None:
        with open(fh_csv, mode='a') as results_csv:
            r_csv_writer = csv.writer(results_csv, delimiter=',')
            if type(metric_teacher_test) is dict:
                # handle ensemble
                for bits, metr in metric_teacher_test.items():
                    name_train, acc_train = metric_teacher_train[bits].get()
                    name_test, acc_test = metr.get()
                    r_csv_writer.writerow([epoch, f"teacher_{bits}", acc_train[0], acc_train[1], acc_test[0], acc_test[1]])
            else:
                name_train, acc_train = metric_teacher_train.get()
                acc_test, acc_test = metric_teacher_test.get()
                r_csv_writer.writerow([epoch, "teacher", acc_train[0], acc_train[1], acc_test[0], acc_test[1]])

    if metric_t_teacher_test is not None:
        with open(fh_csv, mode='a') as results_csv:
            r_csv_writer = csv.writer(results_csv, delimiter=',')
            name_train, acc_train = metric_t_teacher_train.get()
            acc_test, acc_test = metric_t_teacher_test.get()
            r_csv_writer.writerow([epoch, "teacher_t", acc_train[0], acc_train[1], acc_test[0], acc_test[1]])


def log_metrics(logger, phase, metrics, epoch, summary_writer, global_step, sep=": "):
    for bits, metric in metrics.items():
        name, acc = metric.get()
        logger.info(f'[Epoch {epoch}] {phase} Bits {bits} {sep} {name[0]}={acc[0]} {name[1]}={acc[1]}, timestamp: {datetime.now().strftime("%m-%dT%H:%M:%S")}')
        if summary_writer:
            name_0 = f'{name[0]}_bits_{bits}'
            name_1 = f'{name[1]}_bits_{bits}'
            summary_writer.add_scalar("%s/%s" % (name_0, phase), acc[0], global_step=global_step)
            summary_writer.add_scalar("%s/%s" % (name_1, phase), acc[1], global_step=global_step)


def save_current_best_params(logger, net, epoch, best_acc, path):
    net.save_parameters(path)
    logger.info('[Epoch %d] Saving new best to %s with Accuracy: %.4f',
                    epoch, path, best_acc)


def save_checkpoint(opt, logger, net, trainer, epoch, top1, force_save=False):
    # TODO refactor this into regular checkpoitns and best checkpoint
    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0 or force_save:
        fname = os.path.join(opt.prefix, '%s_%sbit_%04d_acc_%.4f.{}' % (opt.model, opt.bits, epoch, top1))
        net.save_parameters(fname.format("params"))
        trainer.save_states(fname.format("states"))
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f',
                    epoch, fname.format("{params,states}"), top1)


def test(model, metric, ctx, val_data, batch_fn, testing=False, teacher_outputs=None, return_output=False, metric_agreement=None, ensemble_teacher=False):

    if ensemble_teacher:
        for bits, metr in metric.items():
            metr.reset()
    else:
        metric.reset()
    if metric_agreement is not None:
        metric_agreement.reset()
    if hasattr(val_data, "reset"):
        val_data.reset()

    all_outputs = []
    for idx, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = []
        # only seems to run once
        for x in data:
            out = model(x)
            if ensemble_teacher:
                outputs_dict = {}
                for (idx, output) in enumerate(out):
                    outputs_dict[model.nets[idx].bits] = output
                out = outputs_dict

            outputs.append(out)

        if ensemble_teacher:
            for idx in range(len(outputs)):
                out_curr = outputs[idx]
                label_curr = label[idx]
                for bits, metr in metric.items():
                    # TODO hotfix when using a single model as ensemble teacher, refactor
                    if bits not in out_curr.keys() and len(out_curr.keys()) == 1:
                        bits = list(out_curr.keys())[0]

                    metr.update(label_curr, out_curr[bits])
        else:
            metric.update(label, outputs)


        if teacher_outputs is not None:
            teacher_out = teacher_outputs[idx]
            if ensemble_teacher:
                teacher_out = teacher_out[model.bits]

            #metric_agreement.update(teacher_out, outputs)
        if return_output:
            out = outputs
            all_outputs.append(out)
        if testing:
            break

    if return_output:
        return metric, all_outputs
    else:
        return metric, metric_agreement

def setup_teacher_trainer(opt, tp, ctx, teacher, t_teacher=False):
    kv = mx.kv.create(opt.kvstore)
    teacher.collect_params().reset_ctx(ctx)

    print("adding warumup epochs to epochs for teacher, warming up without training student")
    opt.epochs = opt.epochs + opt.warmup_epochs

    if t_teacher:
        optimizer_params = OptimizerParams(optimizer=opt.tt_optimizer, wd=opt.tt_wd, momentum=opt.tt_momentum, dtype=opt.dtype)
        lr_params = LRScheduleParams(lr_mode=opt.tt_lr_mode, lr=opt.tt_lr, lr_factor=opt.tt_lr_factor, lr_steps=opt.tt_lr_steps,
                                     warmup_epochs=opt.warmup_epochs, epochs=opt.epochs, dataset=opt.dataset)
    else:
        optimizer_params = OptimizerParams(optimizer=opt.teacher_optimizer, wd=opt.teacher_wd, momentum=opt.teacher_momentum, dtype=opt.dtype)
        lr_params = LRScheduleParams(lr_mode=opt.teacher_lr_mode, lr=opt.teacher_lr, lr_factor=opt.teacher_lr_factor, lr_steps=opt.teacher_lr_steps,
                                 warmup_epochs=opt.warmup_epochs, epochs=opt.epochs, dataset=opt.dataset)

    trainer = gluon.Trainer(teacher.collect_params(), *get_optimizer(optimizer_params, lr_params, tp.batch_size), kvstore=kv)
    return trainer


def setup_training(tp, ctx, model, teacher=None):
    opt = tp.opt
    logger = tp.logger

    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    kv = mx.kv.create(opt.kvstore)
    train_data, val_data, batch_fn = get_data_iters(opt)
    model.collect_params().reset_ctx(ctx)
    optimizer_params = OptimizerParams(optimizer=opt.optimizer, wd=opt.wd, momentum=opt.momentum, dtype=opt.dtype,
                                       activation_method=opt.activation_method, wd_alpha=opt.wd_alpha)
    lr_params = LRScheduleParams(lr_mode=opt.lr_mode, lr=opt.lr, lr_factor=opt.lr_factor, lr_steps=opt.lr_steps,
                                 warmup_epochs=opt.warmup_epochs, epochs=opt.epochs, dataset=opt.dataset)
    trainer = gluon.Trainer(model.collect_params(), *get_optimizer(optimizer_params, lr_params, tp.batch_size), kvstore=kv)

    teacher_trainer = None
    if teacher is not None:
        kv = mx.kv.create(opt.kvstore)
        teacher.collect_params().reset_ctx(ctx)
        if KnowledgeDistiTeacher(tp.opt.kd_teacher_mode) == KnowledgeDistiTeacher.internal:
            # dont create new params for internal, cause teacher uses the ones of the ensemble
            teacher_optimizer_params = optimizer_params
            teacher_lr_params = lr_params
        else:
            teacher_optimizer_params = OptimizerParams(optimizer=opt.teacher_optimizer, wd=opt.teacher_wd,
                                               momentum=opt.teacher_momentum, dtype=opt.dtype)
            teacher_lr_params = LRScheduleParams(lr_mode=opt.teacher_lr_mode, lr=opt.teacher_lr,
                                         lr_factor=opt.teacher_lr_factor, lr_steps=opt.teacher_lr_steps,
                                         warmup_epochs=opt.warmup_epochs, epochs=opt.epochs, dataset=opt.dataset)


    if KnowledgeDistiTeacher(tp.opt.kd_teacher_mode) == KnowledgeDistiTeacher.untrained_external or  KnowledgeDistiTeacher(tp.opt.kd_teacher_mode) == KnowledgeDistiTeacher.internal:
        teacher_trainer = gluon.Trainer(teacher.collect_params(),
                *get_optimizer(teacher_optimizer_params, teacher_lr_params, tp.batch_size), kvstore=kv)

            #teacher_trainer = setup_teacher_trainer(opt, tp, ctx, teacher)


    if opt.resume_states != '':
        trainer.load_states(opt.resume_states)

    summary_writer = None
    if opt.write_summary:
        from mxboard import SummaryWriter
        summary_writer = SummaryWriter(logdir=tp.mxboard_summary, flush_secs=60)
    write_net_summaries(model, summary_writer, ctx[0], write_grads=False)
    track_lr = LRTracker(trainer, summary_writer, tp.logger)

    # set batch norm wd to zero
    params = model.collect_params('.*batchnorm.*')
    for key in params:
        params[key].wd_mult = 0.0

    total_time = 0
    num_epochs = 0
    best_acc = -1
    epoch_time = -1
    num_examples = get_num_examples(opt.dataset)

    return opt, logger, train_data, val_data, batch_fn, trainer, track_lr, total_time, num_epochs, best_acc, epoch_time, num_examples, summary_writer, teacher_trainer


def finish_batch(opt, tp, logger, metrics, epoch, summary_writer, global_step, i, btic, num_examples, tic, epoch_time, track_lr):

    # Returns wether training loop should break
    if opt.log_interval and not (i + 1) % opt.log_interval:
        log_metrics(logger, "batch", metrics, epoch, summary_writer, global_step,
                    sep=" [%d]\tSpeed: %f samples/sec\t" % (i, tp.batch_size / (time.time() - btic)))
        log_progress(num_examples, opt, epoch, i, time.time() - tic, epoch_time)
        track_lr(epoch, global_step)

    global_step += tp.batch_size
    if opt.test_run:
        return True, global_step, btic

    btic = time.time()
    global_step += tp.batch_size
    if opt.test_run:
        return True, global_step, btic

    return False, global_step, btic

def finish_epoch(model, tic, summary_writer, ctx, global_step, num_epochs, total_time, epoch, logger):
    epoch_time = time.time() - tic
    write_net_summaries(model, summary_writer, ctx[0], global_step=global_step)

    # First epoch will usually be much slower than the subsequent epics,
    # so don't factor into the average
    if num_epochs > 0:
        total_time = total_time + epoch_time
    num_epochs = num_epochs + 1

    logger.info('[Epoch %d] time cost: %f' % (epoch, epoch_time))
    if summary_writer:
        summary_writer.add_scalar("training/epoch", epoch, global_step=global_step)
        summary_writer.add_scalar("training/epoch-time", epoch_time, global_step=global_step)

    return total_time, num_epochs


def log_epoch(model, logger, metrics, metrics_test, epoch, summary_writer, global_step, tp, logging, trainer, current_acc,
              best_acc, metric_teacher_train=None, metric_teacher_test=None, agreement_train_metrics=None, agreement_test_metrics=None, teacher=None,
              metric_t_teacher_train=None, metric_t_teacher_test=None, t_teacher=None):
    opt = tp.opt
    log_metrics(logger, f"validation", metrics_test, epoch, summary_writer, global_step)
    if metric_teacher_test is not None:
        if type(metric_teacher_test) is dict:
            log_metrics(logger, f"teacher validation", metric_teacher_test, epoch, summary_writer, global_step)
        else:
            log_metrics(logger, f"teacher validation", {32: metric_teacher_test}, epoch, summary_writer, global_step)

    if metric_t_teacher_test is not None:
        log_metrics(logger, f"t_teacher vaidation", {32: metric_t_teacher_test}, epoch, summary_writer, global_step)
    log_epoch_to_csv(metrics, metrics_test, epoch, tp.fh_csv, metric_teacher_train, metric_teacher_test,
                     agreement_train_metrics, agreement_test_metrics, metric_t_teacher_train, metric_t_teacher_test)

    if opt.interrupt_at is not None and epoch + 1 == opt.interrupt_at:
        logging.info("[Epoch %d] Interrupting run now because 'interrupt-at' was set to %d..." %
                     (epoch, opt.interrupt_at))
        save_checkpoint(opt, logger, model, trainer, epoch, current_acc, force_save=True)
        sys.exit(3)

    # save model if meet requirements
    if best_acc < current_acc:
        best_acc = current_acc
        save_current_best_params(logger, model, epoch, current_acc, tp.best_checkpoint_path)
        if tp.opt.kd_teacher_mode == 'untrained_external':
            suffix = '.params'
            path = tp.best_checkpoint_path.split(suffix)[0]
            path = path + '_teacher' + suffix
            save_current_best_params(logger, teacher, epoch, current_acc, path)
            if tp.opt.teacher_teacher is not '':
                suffix = '.params'
                path = tp.best_checkpoint_path.split(suffix)[0]
                path = path + '_t_teacher' + suffix
                save_current_best_params(logger, t_teacher, epoch, current_acc, path)

    return best_acc