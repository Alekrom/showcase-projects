import argparse
import sys

import models
from models.common_layers import imagenet_variants
from datasets.util import get_all_dataset_names


def set_dummy_training_args(opt):
    dummy_parser = argparse.ArgumentParser()
    training_args(dummy_parser)
    optimizer_args(dummy_parser)
    dummy_opt = dummy_parser.parse_args([])
    for k, v in dummy_opt.__dict__.items():
        setattr(opt, k, v)


def data_args(parser):
    data = parser.add_argument_group('Data', 'parameters regarding datasets')
    data.add_argument('--batch-size', type=int, default=64,
                      help='training batch size per device (CPU/GPU).')
    data.add_argument('--data-dir', type=str, default='',
                      help='training directory of imagenet images, contains train/val subdirs.')
    data.add_argument('--data-path', type=str, default='.',
                      help='training directory where cifar10 / mnist data should be or is saved.')
    data.add_argument('--dataset', type=str, default='cifar10', choices=get_all_dataset_names(),
                      help='dataset to use. options are mnist, cifar10, imagenet and dummy.')
    data.add_argument('--dtype', default='float32', type=str,
                      help='data type, float32 or float16 if applicable')
    data.add_argument('--num-worker', '-j', dest='num_workers', default=4, type=int,
                      help='number of workers of dataloader.')
    data.add_argument('--augmentation', choices=["low", "medium", "high"], default="medium",
                      help='how much augmentation to use')
    data.add_argument('--result_path', default="../results/")


def running_args(parser):
    parser.add_argument('--builtin-profiler', type=int, default=0,
                        help='Enable built-in profiler (0=off, 1=on)')
    parser.add_argument('--gpus', type=str, default='',
                        help='ordinates of gpus to use, can be "0,1,2" or empty for cpu only.')
    parser.add_argument('--cpu', action="store_const", dest="gpus", const="",
                        help='to explicitly use cpu')
    parser.add_argument('--mode', type=str, choices=["symbolic", "imperative", "hybrid"], default="imperative",
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')


def evaluation_args(parser):
    parser.add_argument('--params', type=str, required=True,
                        help='the .params with the model weights')
    parser.add_argument('--verbose', action="store_true",
                        help='prints information about the model before evaluating')
    parser.add_argument('--limit-eval', type=int, metavar="N", default=-1,
                        help="stop evaluation after N iterations")


def optimizer_args(parser):
    optimizer = parser.add_argument_group('Optimizer', 'parameters for optimizers')
    optimizer.add_argument('--lr-mode', type=str, default='cosine',
                           help='learning rate scheduler mode. options are step, poly and cosine.')
    optimizer.add_argument('--lr', type=float, default=0.01,
                           help='learning rate. default is 0.01.')
    optimizer.add_argument('--lr-factor', default=0, type=float,
                           help='learning rate decay ratio')
    optimizer.add_argument('--lr-steps', default='90,100', type=str,
                           help='list of learning rate decay epochs as in str')
    optimizer.add_argument('--momentum', type=float, default=0.9,
                           help='momentum value for optimizer, default is 0.9.')
    optimizer.add_argument('--optimizer', type=str, default="radam",
                           help='the optimizer to use. default is radam.')
    optimizer.add_argument('--wd', type=float, default=0.0,
                           help='weight decay rate. default is 0.0.')
    optimizer.add_argument('--wd-alpha', type=float, default=0.0,
                           help='weight decay rate for parameter alpha of pact activations. default is 0.0.')
    optimizer.add_argument('--warmup-lr', type=float, default=0.0,
                           help='starting warmup learning rate. default is 0.0.')
    optimizer.add_argument('--warmup-epochs', type=int, default=0,
                           help='number of warmup epochs.')


def training_args(parser):
    train = parser.add_argument_group('Training', 'parameters for training')
    train.add_argument('--dry-run', action='store_true',
                       help='do not train, only do other things, e.g. output args and plot network')
    train.add_argument('--test-run', action='store_true',
                       help='only train one batch per epoch for testing the whole training.')
    train.add_argument('--interrupt-at', type=int, default=None,
                       help='interrupt the training at a certain epoch. script will exit mid-training with code 3.')
    train.add_argument('--epochs', type=int, default=120,
                       help='number of training epochs.')
    train.add_argument('--initialization', type=str, choices=["default", "gaussian", "msraprelu_avg", "msraprelu_in"],
                       default="gaussian",
                       help='weight initialization, default is xavier with magnitude 2.')
    train.add_argument('--kvstore', type=str, default='device',
                       help='kvstore to use for trainer/module.')
    train.add_argument('--log', type=str, default='image-classification',
                       help='Filename and path where log file should be stored.')
    train.add_argument('--log-interval', type=int, default=50,
                       help='Number of batches to wait before logging.')
    train.add_argument('--plot-network', type=str, default=None,
                       help='Whether to output the network plot.')
    train.add_argument('--profile', action='store_true',
                       help='Option to turn on memory profiling for front-end, and prints out '
                            'the memory usage by python function at the end.')
    train.add_argument('--progress', type=str, default="",
                       help='save progress and ETA to this file')
    train.add_argument('--resume', type=str, default='',
                       help='path to saved weight where you want resume')
    train.add_argument('--resume-states', type=str, default='',
                       help='path of trainer state to load from.')
    train.add_argument('--save-frequency', default=None, type=int,
                       help='epoch frequence to save model, best model will always be saved')
    train.add_argument('--seed', type=int, default=123,
                       help='random seed to use. Default=123.')
    train.add_argument('--start-epoch', default=0, type=int,
                       help='starting epoch, 0 for fresh training, > 0 to resume')
    train.add_argument('--write-summary', type=str, default=None,
                       help='write tensorboard summaries to this path, use ../mxboard_summary')
    train.add_argument('--loss-mode', type=str, default="equal", choices=["equal"],
                       help='how to calculate the overall loss of the model')
    train.add_argument('--two-stage', action='store_true',
                       help='Apply training in two stages (binarize only activations in first stage')
    train.add_argument('--train-single', default=None, type=int,
                       help='Train only selected bit width')
    train.add_argument('--bit-widths', type=int, nargs='+',
                       help="The bit width to train with the ensemble")
    train.add_argument('--train-teacher', action='store_true',
                    help="train just the teacherteacher")
    train.add_argument('--target-bits', type=int, default=1,
                       help="which bit width to optimize")
    train.add_argument('--fp-weights', action='store_true',
                       help="train network with full precision weights")
    train.add_argument('--identity-clip', action='store_true', default=False,
                       help="use identity and clip for fp model")
    train.add_argument('--no-batch-reuse', action='store_true', default=False,
                       help="use a single batch for training each model in sharedweightnet during forward pass")


def knowledge_distillation_args(parser):
    kd = parser.add_argument_group('Knowledge Distillation', 'parameters for knowledge distillation')
    kd.add_argument('--kd-mode', type=str, default="none", choices=["none", "simple", "progressive"],
                       help="How to apply knowledge distillation")
    kd.add_argument('--kd-teacher-mode', type=str, default="none",
                       choices=["none", "pretrained_external", "untrained_external", "internal"],
                       help="Teacher for knowledge distillation")
    kd.add_argument('--kd-alpha', type=float, default=0.0,
                       help="Balancing parameter for knowledge distillation")
    kd.add_argument('--teacher', type=str, default="",
                       help="model of the teacher from gluon")
    kd.add_argument('--teacher-params', type=str, default="",
                       help="path to parameters of the teacher")
    kd.add_argument('--teacher-lr-mode', type=str, default='cosine',
                           help='learning rate scheduler mode. options are step, poly and cosine.')
    kd.add_argument('--teacher-lr', type=float, default=0.01,
                           help='learning rate. default is 0.01.')
    kd.add_argument('--teacher-lr-factor', default=0.1, type=float,
                           help='learning rate decay ratio')
    kd.add_argument('--teacher-lr-steps', default='30,60,90', type=str,
                           help='list of learning rate decay epochs as in str')
    kd.add_argument('--teacher-momentum', type=float, default=0.9,
                           help='momentum value for optimizer, default is 0.9.')
    kd.add_argument('--teacher-optimizer', type=str, default="adam",
                           help='the optimizer to use. default is adam.')
    kd.add_argument('--teacher-wd', type=float, default=0.0,
                           help='weight decay rate. default is 0.0.')
    kd.add_argument('--teacher-warmup-lr', type=float, default=0.0,
                           help='starting warmup learning rate. default is 0.0.')
    kd.add_argument('--teacher-warmup-epochs', type=int, default=0,
                           help='number of warmup epochs.')
    kd.add_argument('--ensemble-teacher', action='store_true',
                       help="teacher is ensemble")
    kd.add_argument('--attention-matching', type=int, default=0,
                    help="how to apply attention matching, 0: no attention matching, n: use only bit maps of teacher with bit width n for attention matching, -1 use all teacher bit width for attention matching")
    kd.add_argument('--teacher-resume', type=str, default='', help="path to saved weights of teacher model")
    kd.add_argument('--teacher-teacher', type=str, default="",
                    help="model of the teacher from gluon")
    kd.add_argument('--tt-params', type=str, default="",
                    help="path to parameters of the teacher")
    kd.add_argument('--tt-lr-mode', type=str, default='cosine',
                    help='learning rate scheduler mode. options are step, poly and cosine.')
    kd.add_argument('--tt-lr', type=float, default=0.01,
                    help='learning rate. default is 0.01.')
    kd.add_argument('--tt-lr-factor', default=0.1, type=float,
                    help='learning rate decay ratio')
    kd.add_argument('--tt-lr-steps', default='30,60,90', type=str,
                    help='list of learning rate decay epochs as in str')
    kd.add_argument('--tt-momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
    kd.add_argument('--tt-optimizer', type=str, default="adam",
                    help='the optimizer to use. default is adam.')
    kd.add_argument('--tt-wd', type=float, default=0.0,
                    help='weight decay rate. default is 0.0.')
    kd.add_argument('--tt-warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
    kd.add_argument('--tt-warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
    kd.add_argument('--tt-kd-mode', type=str, default="none", choices=["none", "simple", "progressive"],
                    help="How to apply knowledge distillation between teacher and teacher-teacher")


def add_conditional_parameters(parser):
    first_args = sys.argv[1:][:]
    for pattern in ("--help", "-h"):
        if pattern in first_args:
            first_args.remove(pattern)

    if "--model" not in first_args:
        first_args.extend(["--model", "dummy"])

    first_parse, _ = parser.parse_known_args(first_args)

    for model_parameter in models.get_model_parameters():
        model_parameter.add_group(parser, first_parse.model)


def get_parser(training=True):
    all_args = [running_args, data_args]
    if training:
        all_args.extend([training_args, optimizer_args, knowledge_distillation_args])
    else:
        all_args.extend([evaluation_args])

    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    model = parser.add_argument_group('Model', 'parameters for the model definition')
    model.add_argument('--bits', type=int, default=1,
                       help='number of weight bits')
    model.add_argument('--bits-a', type=int, default=1,
                       help='number of bits for activation')
    model.add_argument('--activation-method', type=str, default='det_sign',
                       choices=['identity', 'approx_sign', 'relu', 'clip', 'leaky_clip',
                                'det_sign', 'sign_approx_sign', 'round', 'dorefa', 'pact', 'dorefa_modified'],
                       help='choose activation in QActivation layer')
    model.add_argument('--weight-quantization', type=str, default='det_sign',
                       help='choose weight quantization')
    model.add_argument('--clip-threshold', type=float, default=1.0,
                       help='clipping threshold, default is 1.0.')
    model.add_argument('--add-activation', type=str, default='', choices=['prelu', 'relu', 'elu'],
                       help='choose to add an activation after each quantized/binarized block')
    model.add_argument('--model', type=str, required=True,
                       help='type of model to use. see vision_model for options.')
    model.add_argument('--use-pretrained', action='store_true',
                       help='enable using pretrained model from gluon.')
    model.add_argument('--approximation', type=str, default='',
                       choices=['', 'xnor', 'binet', 'abc'],
                       help='Add a kind of convolution block approximation.')
    model.add_argument('--initial-layers', type=str, default='imagenet', choices=imagenet_variants,
                       help='choose initial model layers for imagenet models')
    parser.add_argument('--prefix', default='', type=str,
                        help='path to checkpoint prefix, default is current working dir')

    for add_args_fn in all_args:
        add_args_fn(parser)

    add_conditional_parameters(parser)

    return parser
