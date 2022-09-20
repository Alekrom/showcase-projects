#import models

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
from attention_matching.attention_matching import AttentionMatchingBinaryConvolution

"""
    Create the ensemble with shared weights for fiven bit widths
"""


def get_base_net(opt, ctx, bin_l_config_p, collector):
    """Model initialization."""
    kwargs = {'ctx': ctx, 'classes': get_num_classes(opt.dataset), "collector": collector}
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

    weight_quantization = opt.weight_quantization
    activation = opt.activation_method
    if opt.identity_clip and bin_l_config_p.bits == 32:
        weight_quantization = 'identity'
        activation = 'clip'

    if bin_l_config_p.activation == 'pact':
        pact_acti_builder = PactActivationBuilder(bin_l_config_p.bits_a, True)
        with gluon.nn.set_binary_layer_config(bits=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a,
                                              approximation=bin_l_config_p.approximation,
                                              grad_cancel=bin_l_config_p.grad_cancel,
                                              activation=activation,
                                              weight_quantization=weight_quantization,
                                              custom_activation=pact_acti_builder.get_pact,
                                              attention_matching_student=bin_l_config_p.attention_matching_student,
                                              post_block_activation=bin_l_config_p.post_block_activation):
            net = models.get_model(opt.model, **kwargs)
    else:
        with gluon.nn.set_binary_layer_config(bits=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a,
                                              approximation=bin_l_config_p.approximation,
                                              grad_cancel=bin_l_config_p.grad_cancel,
                                              activation=activation,
                                              weight_quantization=weight_quantization,
                                              custom_activation=None,
                                              attention_matching_student=bin_l_config_p.attention_matching_student,
                                              post_block_activation=bin_l_config_p.post_block_activation):
            net = models.get_model(opt.model, **kwargs)

    if opt.mode != 'symbolic':
        net.cast(opt.dtype)

    return net


class SharedWeightNetwork(HybridBlock):

    def __init__(self, opt, ctx, bit_widths, logger, fp_weights=False, attention_matching=0, attention_matching_student=False):
        super(SharedWeightNetwork, self).__init__()
        self.opt = opt
        self.ctx = ctx
        self.bit_widths = sorted(bit_widths)
        self.nets = []
        collector = PropertyCollector()
        # TODO maybe remove logger
        self.logger = logger

        # Choose max bit width as base model
        max_bit = max(bit_widths)

        if fp_weights:
            bin_l_config_p = BinaryLayerConfigParams(bits=32, bits_a=max_bit,
                                                     weight_quantization='identity',
                                                     activation=opt.activation_method,
                                                     approximation=opt.approximation,
                                                     grad_cancel=opt.clip_threshold,
                                                     post_block_activation=opt.add_activation,
                                                     attention_matching_student=False)
        else:
            bin_l_config_p = BinaryLayerConfigParams(bits=max_bit, bits_a=max_bit,
                                                     weight_quantization=opt.weight_quantization,
                                                     activation=opt.activation_method,
                                                     approximation=opt.approximation,
                                                     grad_cancel=opt.clip_threshold,
                                                     post_block_activation=opt.add_activation,
                                                     attention_matching_student=attention_matching_student)

        #self.logger.info("layer config")
        #self.logger.info(bin_l_config_p.describe())

        # set customconv for attention matching
        if self.should_use_att_matching_conv(attention_matching, bin_l_config_p, attention_matching_student):
            gluon.nn.activated_conv.custom_conv = AttentionMatchingBinaryConvolution
        base_net = get_base_net(self.opt, self.ctx, bin_l_config_p, collector)

        gluon.nn.activated_conv.custom_conv = None

        self.register_child(base_net)
        self.nets.append(base_net)

        # build other nets
        builder = NetworkRebuilder(base_net, collector)
        for bits in bit_widths:
            # skip max bit width so its not done twice
            if bits == max_bit:
                continue
            if fp_weights:
                bin_l_config_p.bits = max_bit
            else:
                bin_l_config_p.bits = bits
            bin_l_config_p.bits_a = bits

            if self.should_use_att_matching_conv(attention_matching, bin_l_config_p, attention_matching_student):
                gluon.nn.activated_conv.custom_conv = AttentionMatchingBinaryConvolution

            net = self.build_shared_weight_network(builder, bin_l_config_p)

            gluon.nn.activated_conv.custom_conv = None
            self.nets.append(net)


    def hybrid_forward(self, F, x):
        outputs = None
        for net in self.nets:
            if outputs is None:
                outputs = net(x)
                outputs = F.expand_dims(outputs, axis=0)
            else:
                output = net(x)
                #F.expand.dims(output, axis=0)
                output = F.expand_dims(output, axis=0)
                outputs = F.concat(outputs,output, dim=0)
        return outputs

    def build_shared_weight_network(self, builder, bin_l_config_p):

        if bin_l_config_p.activation == 'pact':
            pact_acti_builder = PactActivationBuilder(bin_l_config_p.bits_a, True)
            with gluon.nn.set_binary_layer_config(bits=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a,
                                              approximation=bin_l_config_p.approximation,
                                              grad_cancel=bin_l_config_p.grad_cancel,
                                              activation=bin_l_config_p.activation,
                                              weight_quantization=bin_l_config_p.weight_quantization,
                                              custom_activation=pact_acti_builder.get_pact,
                                              attention_matching_student=bin_l_config_p.attention_matching_student,
                                              post_block_activation=bin_l_config_p.post_block_activation):
                net = builder.rebuild_net(bits_w=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a)
        else:
            with gluon.nn.set_binary_layer_config(bits=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a,
                                              approximation=bin_l_config_p.approximation,
                                              grad_cancel=bin_l_config_p.grad_cancel,
                                              activation=bin_l_config_p.activation,
                                              weight_quantization=bin_l_config_p.weight_quantization,
                                              custom_activation=None,
                                                  attention_matching_student=bin_l_config_p.attention_matching_student,
                                                  post_block_activation=bin_l_config_p.post_block_activation):
                net = builder.rebuild_net(bits_w=bin_l_config_p.bits, bits_a=bin_l_config_p.bits_a)

        self.register_child(net)
        return net

    def get_networks(self):
        nets = []
        for pair in self.nets:
            nets.append(pair.net)
        return nets


    def initialize_net(self):
        self.initialize(self.get_initializer(), ctx=self.ctx, force_reinit=False)
        print(self.ctx)
        # dummy forward pass to initialize binary layers
        data, _ = self.get_dummy_data(self.opt, self.ctx[0])
        _ = self(data)

    def get_dummy_data(self, opt, ctx):
        data_shape = get_shape(opt)
        shapes = ((1,) + data_shape[1:], (1,))
        return [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in shapes]

    def get_initializer(self):
        main_init = None
        if self.opt.initialization == "default":
            main_init = mx.init.Xavier(magnitude=2)
        if self.opt.initialization == "gaussian":
            main_init = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)
        if self.opt.initialization == "msraprelu_avg":
            main_init = mx.init.MSRAPrelu()
        if self.opt.initialization == "msraprelu_in":
            main_init = mx.init.MSRAPrelu(factor_type="in")
        return main_init

    def get_net_with_bits(self, bits):
        for net in self.nets:
            if net.bits == bits:
                return net

    def get_teacher_net_and_remove(self):
        net = self.get_max_bit_net()
        self.remove_net_with_bits(net.bits)
        return net

    def get_max_bit_net(self):
        max_bits = max(self.bit_widths)
        return self.get_net_with_bits(max_bits)

    def remove_net_with_bits(self, bits):
        net = self.get_net_with_bits(bits)
        self.nets.remove(net)
        self.bit_widths.remove(bits)

    def should_use_att_matching_conv(self, attention_matching, bin_l_config_p, attention_matching_student):
        # use attention mathcing if all bit width shall be used, if its the correct bit width of a student (since all student layers should receive guidance)
        if attention_matching != 0 and (attention_matching == -1 or attention_matching == bin_l_config_p.bits_a or attention_matching_student):
            return True
        else:
            return False