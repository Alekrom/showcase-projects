from __future__ import division

import os

import logging
import sys

# If the python binding doesnt work mxnet can be added manually
#sys.path.insert(1, '/work/mxnet/python/')

import mxnet as mx
import numpy as np
from graphviz import ExecutableNotFound
from mxnet import profiler
from mxnet import gluon
import time
import csv
from multiprocessing import cpu_count
from datetime import datetime
from util.log_progress import log_progress
from gluoncv.utils import LRScheduler, LRSequential
from mxnet import autograd
from mxnet.visualization import print_summary
from datasets.util import *
from util.arg_parser import get_parser
from shared_weight_network import SharedWeightNetwork, get_base_net
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from loss import LossCalculator, ModelLoss, KnowledgeDistiMode, KnowledgeDistiTeacher
from single_net import get_single_model, initialize_net
from training import train_single, train_ensemble
from training.util import TrainingParams
from util.util import BinaryLayerConfigParams
from property_collector import PropertyCollector
from attention_matching.attention_matching import init_output_dicts, attention_matching_mode

from mxnet import nd
import gluoncv

from datasets.util import get_data_iters
from analysis.model_profile import model_profiling

def set_wd_mult(net):
    # Set wd mult to zero for parameters != alpha if pact is used
    # TODO change alpha lr mul back
    if opt.activation_method == 'pact':
        params_others = net.collect_params(".*weight|.*bias")
        params_others.setattr('wd_mult', 0)
        params_alpha = net.collect_params(".*alpha")
        params_alpha.setattr('lr_mult', 0.5)


def load_best_prev_model(param_path):
    net = SharedWeightNetwork(opt, context, bit_widths, logger, opt.two_stage, opt.attention_matching)
    net.load_parameters(param_path, ctx=context)
    return net


def load_teacher_from_prev_model(param_path, teacher):
    # load parameters of prev best checkpoint into teacher
    param_path = param_path.replace(".params", "_teacher.params")
    teacher.load_parameters(param_path, ctx=context)
    return teacher


def get_dummy_data(opt, ctx):
    data_shape = get_shape(opt)
    shapes = ((1,) + data_shape[1:], (1,))
    return [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in shapes]


def test(model, metric, ctx, val_data, batch_fn, testing=False):
    metric.reset()
    if hasattr(val_data, "reset"):
        val_data.reset()
    for batch in val_data:
        data, label = batch_fn(batch, ctx)
        outputs = []
        for x in data:
            outputs.append(model(x))
        metric.update(label[0], outputs[0])
        if testing:
            break
    return metric


class LRTracker:
    def __init__(self, trainer, summary_writer):
        self.trainer = trainer
        self.prev_lr = trainer.learning_rate
        self.summary_writer = summary_writer

    def __call__(self, epoch, global_step=0):
        current_lr = self.trainer.learning_rate
        if current_lr != self.prev_lr:
            logger.info('[Epoch %d] Change learning rate to %f', epoch, current_lr)
            self.prev_lr = current_lr
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("training/lr", current_lr, global_step=global_step)


def plot_net(net, bit, name='network_plot'):
    net.hybridize()
    data, _ = get_dummy_data(opt, context[0])
    graph = mx.viz.plot_network(net(mx.sym.var('data'))[0], node_attrs={"shape": "oval", "fixedsize": "false"})
    graph.format = 'pdf'
    graph.render(name)
    x = mx.sym.var('data')
    sym = net(x)
    print_summary(sym, shape={"data": get_shape(opt)}, quantized_bitwidth=bit)


def get_initializer(opt):
    main_init = None
    if opt.initialization == "default":
        main_init = mx.init.Xavier(magnitude=2)
    if opt.initialization == "gaussian":
        main_init = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)
    if opt.initialization == "msraprelu_avg":
        main_init = mx.init.MSRAPrelu()
    if opt.initialization == "msraprelu_in":
        main_init = mx.init.MSRAPrelu(factor_type="in")
    return main_init


def get_dummy_data(opt, ctx):
    data_shape = get_shape(opt)
    shapes = ((1,) + data_shape[1:], (1,))
    return [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in shapes]


def init_teacher_params(teacher, teacher_teacher=False):
    # load params if pretrained or initialize
    if KnowledgeDistiTeacher(opt.kd_teacher_mode) == KnowledgeDistiTeacher.pretrained_external:
        if teacher_teacher:
            teacher.load_parameters(f'../teacher_params/{opt.tt_params}', ctx=context)
        else:
            teacher.load_parameters(f'../teacher_params/{opt.teacher_params}', ctx=context)
        # Freeze weights for pretrained teacher
        params = teacher.collect_params()
        params.setattr('grad_req', 'null')
    elif KnowledgeDistiTeacher(opt.kd_teacher_mode) == KnowledgeDistiTeacher.untrained_external:
        teacher.initialize(get_initializer(opt), ctx=context, force_reinit=False)
        data, _ = get_dummy_data(opt, context[0])
        _ = teacher(data)

    if opt.mode == 'hybrid':
        teacher.hybridize()
    return teacher


def get_teacher_cifar100():
    teacher = None
    if KnowledgeDistiTeacher(opt.kd_teacher_mode) != KnowledgeDistiTeacher.none and KnowledgeDistiTeacher(opt.kd_teacher_mode) != KnowledgeDistiTeacher.internal:
        if opt.teacher == 'resnet_e1':
            bin_l_config = BinaryLayerConfigParams(bits=32, bits_a=32,
                                    weight_quantization='identity',
                                    activation='clip',
                                    approximation=opt.approximation,
                                    grad_cancel=1.3,
                                    post_block_activation=opt.add_activation)

            teacher = get_base_net(opt, context, bin_l_config, PropertyCollector())
        elif opt.ensemble_teacher:
            teacher = SharedWeightNetwork(opt, context, bit_widths, logger, True, opt.attention_matching, False)
            if KnowledgeDistiTeacher(opt.kd_teacher_mode) != KnowledgeDistiTeacher.untrained_external:
                params = teacher.collect_params()
                params.setattr('grad_req', 'null')
        else:
            teacher = gluoncv.model_zoo.get_model(opt.teacher, classes=get_num_classes(opt.dataset))

        teacher = init_teacher_params(teacher)
    return teacher


def get_teacher_imagenet():
    teacher = None
    if KnowledgeDistiTeacher(opt.kd_teacher_mode) != KnowledgeDistiTeacher.none and KnowledgeDistiTeacher(
            opt.kd_teacher_mode) != KnowledgeDistiTeacher.internal:
        if opt.ensemble_teacher:
            teacher = SharedWeightNetwork(opt, context, bit_widths, logger, True, opt.attention_matching, False)
            teacher.load_parameters(f'../teacher_params/{opt.teacher_params}', ctx=context)
        else:
            pretrained = True if KnowledgeDistiTeacher(opt.kd_teacher_mode) == KnowledgeDistiTeacher.pretrained_external else False
            teacher = gluoncv.model_zoo.get_model(opt.teacher, pretrained=pretrained, classes=get_num_classes(opt.dataset), ctx=context)
            if opt.teacher_resume != '':
                teacher.load_parameters(f'{opt.teacher_resume}', ctx=context)
        teacher.initialize(get_initializer(opt), ctx=context, force_reinit=False)
        data, _ = get_dummy_data(opt, context[0])
        _ = teacher(data)
    return teacher


def get_teacher_teacher():
    teacher = None
    if opt.teacher_teacher != '':
        if KnowledgeDistiTeacher(opt.kd_teacher_mode) != KnowledgeDistiTeacher.none and KnowledgeDistiTeacher(opt.kd_teacher_mode) != KnowledgeDistiTeacher.internal:
            teacher = gluoncv.model_zoo.get_model(opt.teacher_teacher, classes=get_num_classes(opt.dataset))

        teacher = init_teacher_params(teacher, True)
    return teacher


if __name__ == '__main__':
    print(f"CPU COUNT: {cpu_count()}")
    parser = get_parser()
    opt = parser.parse_args()
    result_path = opt.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # logging
    logging.basicConfig(level=logging.INFO)
    log_prefix = opt.log
    datetime_now = datetime.now(tz=None)
    fh_name = f'{log_prefix}{datetime_now}'
    fh_log = logging.FileHandler(f'{result_path}/{fh_name}.log'.replace(' ', '_'))
    logger = logging.getLogger()
    logger.addHandler(fh_log)
    formatter = logging.Formatter('%(message)s')
    fh_log.setFormatter(formatter)
    fh_log.setLevel(logging.DEBUG)
    logging.debug('\n%s', '-' * 100)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh_log.setFormatter(formatter)

    fh_csv = f'{result_path}/{fh_name}.csv'.replace(' ', '_')

    with open(fh_csv, mode='a') as results_csv:
        r_csv_writer = csv.writer(results_csv, delimiter=',')
        r_csv_writer.writerow(['epoch', 'bits', 'top_acc_train', 'top5_acc_train', 'top_acc_test', 'top5_acc_test',
                               'top_agreement_train', 'kl_agreement_train', 'top_agreement_test', 'kl_agreement_test'])

    if opt.write_summary:
        mxboard_summary = f'{opt.write_summary}/{log_prefix}{datetime_now}'.replace(' ', '_')
    else:
        mxboard_summary = None

    # global variables
    if opt.save_frequency is None:
        opt.save_frequency = get_default_save_frequency(opt.dataset)
    logger.info('Starting new image-classification task:, %s', opt)

    mx.random.seed(opt.seed)

    best_checkpoint_path = os.path.join(opt.prefix, f'{result_path}{fh_name}_best_params.params'.replace(' ', '_'))

    test_image_path = '/home/alexander/testimages/'

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    batch_size, dataset, classes = opt.batch_size, opt.dataset, get_num_classes(opt.dataset)
    context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
    if opt.dry_run:
        context = [mx.cpu()]
    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    opt.batch_size = batch_size

    print(f"context {context}")
    print(f"batchsize {batch_size}")

    bit_widths = opt.bit_widths
    if opt.attention_matching != 0:
        init_output_dicts()
        attention_matching_mode = 0


    if opt.train_single is None:
        loss_calculator = LossCalculator(loss_fn, len(bit_widths), opt.loss_mode, opt.kd_mode, opt.kd_teacher_mode, opt.kd_alpha, opt.tt_kd_mode)
    else:
        loss_calculator = LossCalculator(loss_fn, 1, opt.loss_mode, opt.kd_mode, opt.kd_teacher_mode, opt.kd_alpha, opt.tt_kd_mode)

    target_bits = opt.target_bits
    if target_bits == 1 and target_bits not in bit_widths:
        target_bits = 4
    tp = TrainingParams(opt, batch_size, logger, loss_fn, loss_calculator, fh_csv, best_checkpoint_path, mxboard_summary, target_bits, opt.no_batch_reuse)

    #TODO adjust teacher_teacher
    t_teacher = get_teacher_teacher()

    if opt.dataset == 'cifar100':
        teacher = get_teacher_cifar100()
    else:
        teacher = get_teacher_imagenet()


    if opt.train_teacher:
        teacher = gluoncv.model_zoo.get_model(opt.teacher, classes=get_num_classes(dataset))
        teacher.initialize(get_initializer(opt), ctx=context, force_reinit=False)
        data, _ = get_dummy_data(opt, context[0])
        _ = teacher(data)
        setattr(teacher, "bits", 32)
        train_single.train(teacher, tp, context)
    else:
        # real training
        fp_weights = opt.two_stage or opt.fp_weights
        shared_weight_net = SharedWeightNetwork(opt, context, bit_widths, logger, fp_weights,  opt.attention_matching, True)
        if opt.resume == '':
            shared_weight_net.initialize_net()
        else:
            shared_weight_net.load_parameters(opt.resume, ctx=context)

        if KnowledgeDistiTeacher(opt.kd_teacher_mode) == KnowledgeDistiTeacher.internal:
            # get highest bit net as teacher
            teacher = shared_weight_net.get_teacher_net_and_remove()

        if opt.mode == 'hybrid':
            shared_weight_net.hybridize()
        # fix weight decay for pact
        #set_wd_mult(shared_weight_net)

        # model plotting and analysis
        #plot_net(shared_weight_net, 32, name='description_ensemble')
        #for bit in shared_weight_net.bit_widths:
        #    plot_net(shared_weight_net.get_net_with_bits(bit), bit, name=f'single net {bit}')
        #plot_net(teacher, name='testtest')
        #data, _ = get_dummy_data(opt, context[0])

        #profile_dict = dict()
        #for bit in shared_weight_net.bit_widths:
        #    profile_dict[bit] = list(model_profiling(shared_weight_net.get_net_with_bits(bit), data, (bit, bit)))

        train_ensemble.train(shared_weight_net, tp, context, teacher, t_teacher, opt.ensemble_teacher)

    # retrain best model if two stage
    if opt.two_stage:
        # fp_weights_path is best checkpoint path with fpweights in name
        fp_weights_path = os.path.join(opt.prefix, f'{result_path}{fh_name}_fpweights_best_params.params'.replace(' ', '_'))
        shared_weight_net.save_parameters(fp_weights_path)
        opt.two_stage = False
        opt.kd_teacher_mode = "pretrained_external"
        opt.teacher_ensemble = True
        loss_calculator = LossCalculator(loss_fn, len(bit_widths), opt.loss_mode, opt.kd_mode, opt.kd_teacher_mode,
                                         opt.kd_alpha)
        tp.loss_calculator = loss_calculator
        logger.info('Starting training second stage')

        shared_weight_net = load_best_prev_model(best_checkpoint_path)

        # freeze teacher since its already trained
        teacher = load_teacher_from_prev_model(best_checkpoint_path, teacher)
        teacher.collect_params().setattr('grad_req', 'null')

        train_ensemble.train(shared_weight_net, tp, context, teacher, t_teacher, opt.ensemble_teacher)