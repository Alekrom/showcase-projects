"""
Adopted from
@article{DBLP:journals/corr/abs-1912-09666,
  author    = {Qing Jin and
               Linjie Yang and
               Zhenyu Liao},
  title     = {AdaBits: Neural Network Quantization with Adaptive Bit-Widths},
  journal   = {CoRR},
  volume    = {abs/1912.09666},
  year      = {2019},
  url       = {http://arxiv.org/abs/1912.09666},
  eprinttype = {arXiv},
  eprint    = {1912.09666},
  timestamp = {Fri, 03 Jan 2020 16:10:45 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1912-09666.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

import numpy as np
import time
import mxnet
import mxnet.gluon.nn as nn
import pprint



model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
bytes_space = 15
bitops_space = 15
energy_space = 15
latency_space = 15
seconds_space = 15

num_forwards = 10

switch_alpha = False
len_bitlist = 5
bits_curr = (32, 32)

total_params = 0
total_macs = 0
total_bitops = 0



def get_params(self, input):
    """get number of params in module"""
    #return np.sum(
    #    [np.prod(list(w.shape)) for w in self.collect_params()])
    summary_dict = self.get_params(input[0])
    pprint.pprint(summary_dict, sort_dicts=False)
    layer = summary_dict.popitem(last=False)
    print(layer['n_params'])
    return layer['n_params']


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, verbose):
    global total_macs, total_params, total_bitops
    ins = input[0].shape
    outs = output.shape
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(self)
    if isinstance(self, nn.Conv2D) or isinstance(self, nn.QConv2D) or isinstance(self, nn.BinaryConvolution):
        if isinstance(self, nn.BinaryConvolution):
            return
        #    print("binary")
        #    print(self.name)
        #    self = self.qconv
        kernel_size = self._kwargs['kernel']
        groups = self._kwargs['num_group']
        self.n_macs = (ins[1] * outs[1] *
                       kernel_size[0] * kernel_size[1] *
                       outs[2] * outs[3] // groups) * outs[0]
        #self.n_params = get_params(self)
        self.n_params = ins[1] * outs[1] * kernel_size[0] * kernel_size[1] // groups
        if self.bias is not None:
            self.n_params += outs[1]
        bitw, bita = bits_curr

        self.n_bitops = self.n_macs * bitw * bita * 1e-9

        self.n_bytes = (ins[1] * outs[1] * kernel_size[0] * kernel_size[1] // groups) * bitw / 8e6
        if self.bias is not None:
            self.n_bytes += outs[1] * 4e-6
        if getattr(self, 'alpha', None) is not None:
            if switch_alpha is False:
                self.n_bytes += 4e-6 * len_bitlist
            else:
                self.n_bytes += 4e-6
        self.energy = 0
        self.latency = 0
        self.n_seconds = 0

        total_macs += self.n_macs
        total_params += self.n_params
        total_bitops += self.n_bitops

        #self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Dense):
        self.n_macs = ins[1] * outs[1] * outs[0]
        #self.n_params = get_params(self)
        self.n_params = ins[1] * outs[1]
        if self.bias is not None:
            self.n_params += outs[1]
        bitw, bita = getattr(self, 'get_bits', lambda: [32, 32])()
        self.n_bitops = self.n_macs * 1e-9
        self.n_bytes = (ins[1] * outs[1]) * bitw / 8e6
        if self.bias is not None:
            self.n_bytes += outs[1] * bitw * bita * 4e-6
        if getattr(self, 'alpha', None) is not None:
            if switch_alpha is False:
                self.n_bytes += 4e-6 * len_bitlist
            else:
                self.n_bytes += 4e-6

        self.energy = 0
        self.latency = 0
        self.n_seconds = 0
        #self.name = self.__repr__()
        total_macs += self.n_macs
        total_params += self.n_params
        total_bitops += self.n_bitops
    elif isinstance(self, nn.AvgPool2D):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_bitops = self.n_macs * 32 * 32 * 1e-9
        self.n_bytes = 0
        self.energy = 0
        self.latency = 0
        self.n_seconds = 0
        #self.name = self.__repr__()
        total_macs += self.n_macs
        total_params += self.n_params
        total_bitops += self.n_bitops
    elif isinstance(self, nn.BatchNorm):
        #self.n_macs = 0
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        #self.n_params = get_params(self, input)
        self.n_params = 0
        #if self.weight is not None:
        #    self.n_params += self.num_features
        #if self.bias is not None:
        #    self.n_params += self.num_features
        self.n_bitops = self.n_macs * 32 * 32 * 1e-9
        self.n_bytes = 4e-6 * self.n_params
        self.energy = 0
        self.latency = 0
        self.n_seconds = 0
        #self.name = self.__repr__()
        total_macs += self.n_macs
        total_params += self.n_params
        total_bitops += self.n_bitops
    else:
        if isinstance(self, nn.QActivation):
            return
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_bitops = 0
        self.n_bytes = 0
        self.energy = 0
        self.latency = 0
        self.n_seconds = 0
        num_children = 0
        for m in self._children:
            #print('parent: {}\n(n_macs: {})\nchild: {}\n(n_macs: {})'.format(self, getattr(self, 'n_macs', None), m, getattr(m, 'n_macs', None)))
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_bitops += getattr(m, 'n_bitops', 0)
            self.n_bytes += getattr(m, 'n_bytes', 0)
            self.energy += getattr(m, 'energy', 0)
            self.latency += getattr(m, 'latency', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm, nn.Dropout, nn.Sequential, nn.MaxPool2D,
        ]
        #if (not getattr(self, 'ignore_model_profiling', False) and
        #        self.n_macs == 0 and
        #        t not in ignore_zeros_t):
        #    print(
        #        'WARNING: leaf module {} has zero n_macs.'.format(type(self)))
        return
    if verbose:
        print(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:.2f}'.format(self.n_bytes).rjust(bytes_space, ' ') +
            '{:.2f}'.format(self.n_bitops).rjust(bitops_space, ' ') +
            '{:.2f}'.format(self.energy).rjust(energy_space, ' ') +
            '{:.2f}'.format(self.latency).rjust(latency_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
      m.register_forward_hook(
        lambda m, input, output: module_profiling(
          m, input, output, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.detach()
    model_profiling_hooks = []


def model_profiling(model, data, bits, batch=1, channel=3, use_cuda=True,
                    verbose=True):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).
    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool
    Returns:
        macs: int
        params: int
    """
    global bits_curr, total_macs, total_params, total_bitops, first_conv
    bits_curr = bits
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose))
    if verbose is not None:
        print(
            'Item'.ljust(name_space, ' ') +
            'params'.rjust(params_space, ' ') +
            'macs'.rjust(macs_space, ' ') +
            'bytes (MB)'.rjust(bytes_space, ' ') +
            'bitops (B)'.rjust(bitops_space, ' ') +
            'energy (mJ)'.rjust(energy_space, ' ') +
            'latency (ms)'.rjust(latency_space, ' ') +
            'nanosecs'.rjust(seconds_space, ' '))
    if verbose:
        print(''.center(name_space+params_space+macs_space+bytes_space+bitops_space+energy_space+latency_space+seconds_space, '-'))
    model(data)
    if verbose:
        print(''.center(name_space+params_space+macs_space+bytes_space+bitops_space+energy_space+latency_space+seconds_space, '-'))
    if verbose is not None:
        print(
            'Total'.ljust(name_space, ' ') +
            '{:,}'.format(model.n_params).rjust(params_space, ' ') +
            '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
            '{:.2f}'.format(model.n_bytes).rjust(bytes_space, ' ') +
            '{:.2f}'.format(model.n_bitops).rjust(bitops_space, ' ') +
            '{:.2f}'.format(model.energy).rjust(energy_space, ' ') +
            '{:.2f}'.format(model.latency).rjust(latency_space, ' ') +
            '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()

    print(f'total_params: {total_params}')
    print(f'total_macs: {total_macs}')
    print(f'total_bitops: {total_bitops}')

    total_params = 0
    total_bitops = 0
    total_macs = 0

    return model.n_macs, model.n_params, model.n_bitops, model.n_bytes