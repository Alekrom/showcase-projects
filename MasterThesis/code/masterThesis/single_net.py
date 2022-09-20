import mxnet as mx
from mxnet import gluon
from mxnet.gluon.block import HybridBlock
from mxnet import nd
import numpy as np
from network_rebuilder import NetworkRebuilder
from property_collector import PropertyCollector
from util.util import BinaryLayerConfigParams
import models
from datasets.util import get_num_classes, get_shape
from models.pact import PactActivationBuilder

def get_single_model(bits, opt, ctx, logger):
    bin_l_config_p = BinaryLayerConfigParams(bits=bits, bits_a=bits,
                                             weight_quantization=opt.weight_quantization,
                                             activation=opt.activation_method,
                                             approximation=opt.approximation,
                                             post_block_activation=opt.add_activation,
                                             grad_cancel=opt.clip_threshold)

    logger.info("layer config")
    logger.info(bin_l_config_p.describe())
    """Model initialization."""
    kwargs = {'ctx': ctx, 'classes': get_num_classes(opt.dataset)}
    if opt.model.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm

    thumbnail_models = ['resnet', 'binet', 'densenet', 'meliusnet']
    if any(opt.model.startswith(name) for name in thumbnail_models) and get_shape(opt)[2] < 50:
        kwargs['initial_layers'] = "thumbnail"
    else:
        kwargs['initial_layers'] = opt.initial_layers

    kwargs['bits'] = bin_l_config_p.bits

    for model_parameter in models.get_model_parameters():
        model_parameter.set_args_for_model(opt, kwargs)

    if bin_l_config_p.activation == 'pact':
        pact_acti_builder = PactActivationBuilder(bin_l_config_p.bits_a, False)
        with gluon.nn.set_binary_layer_config(bits=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a,
                                              approximation=bin_l_config_p.approximation,
                                              grad_cancel=bin_l_config_p.grad_cancel,
                                              activation=bin_l_config_p.activation,
                                              weight_quantization=bin_l_config_p.weight_quantization,
                                              custom_activation=pact_acti_builder.get_pact,
                                              attention_matching_student=bin_l_config_p.attention_matching_student):
            net = models.get_model(opt.model, **kwargs)
    else:
        with gluon.nn.set_binary_layer_config(bits=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a,
                                              approximation=bin_l_config_p.approximation,
                                              grad_cancel=bin_l_config_p.grad_cancel,
                                              activation=bin_l_config_p.activation,
                                              weight_quantization=bin_l_config_p.weight_quantization,
                                              custom_activation=None,
                                              attention_matching_student=bin_l_config_p.attention_matching_student):
            net = models.get_model(opt.model, **kwargs)
    return net

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


def initialize_net(net, opt, ctx):
    net.initialize(get_initializer(opt), ctx=ctx, force_reinit=False)
    print(ctx)
    # dummy forward pass to initialize binary layers
    data, _ = get_dummy_data(opt, ctx[0])
    _ = net(data)