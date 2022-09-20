from mxnet.gluon.nn.binary_layers import BinaryConvolution
from mxnet.gluon.block import HybridBlock
from mxnet import sym, nd
from mxnet.gluon import nn
import mxnet
import re
import string

"""
    Calculate loss for attention matching
    Functions for calculating the loss (calculate_attention_loss_for_layer, normalized_attention_vector, l2_norm_unit_vector, calculate_Q) are adopted from:
    
    larq, Larq Zoo, (2021), GitHub repository, https://github.com/larq/zoo/tree/main/larq_zoo, file: /training/knowledge_distillation/knowledge_distillation.py

"""

# 0: no attention matching, n: use only bit maps of teacher with bit width n for attention matching, -1 use all teacher bit width for attention matching
attention_matching_mode = 0
num_stages_per_resnet = 4

convOutputsDictTeacher = dict()
convOutputsDictStudent = dict()
layer_names = set()

def get_teacher_output_dict():
    return convOutputsDictTeacher

def get_student_output_dict():
    return convOutputsDictStudent

def init_output_dicts():
    teacher = get_teacher_output_dict()
    student = get_student_output_dict()
    teacher = dict()
    student = dict()


def calculate_Q(A):
    return nd.sum(nd.square(A))

def l2_norm_unit_vector(Q):
    norm = nd.L2Normalization(Q)
    unit_vector = Q / norm
    return unit_vector


def normalized_attention_vector(output):
    output_squared = nd.square(output)
    attention_area = nd.mean(output_squared, -1)
    batch_size = attention_area.shape[0]
    attention_vector = nd.reshape(attention_area, (batch_size, -1))
    attention_vector_reduced_max = nd.max(attention_vector, axis=-1, keepdims=True)
    attention_vector_normalized = attention_vector / attention_vector_reduced_max
    return attention_vector_normalized

def calculate_attention_loss_for_layer(layer_name):
    student_output = convOutputsDictStudent[layer_name]
    if attention_matching_mode == -1:
        teacher_output = convOutputsDictTeacher[layer_name].detach()
    else:
        # match student layer to teacher layer, in sharednet, stage index in name of conv layer is increamented for each stage in ensemble. since all models have the same number of stages:
        # stageindex % number of stages in a resnete uniquely identifies position of stage in model and enables mapping to teacher stage
        stage_idx_search = re.search(r'stage(\d+)', layer_name)
        stage_idx = int(stage_idx_search.group(1))
        stage_idx_teacher = stage_idx % num_stages_per_resnet
        teacher_layer_name = re.sub(r"stage(\d+)", f"stage{str(stage_idx_teacher)}", layer_name)
        teacher_output = convOutputsDictTeacher[teacher_layer_name].detach()

    teacher_norm = normalized_attention_vector(teacher_output)
    student_norm = normalized_attention_vector(student_output)
    loss = nd.mean(nd.square(student_norm - teacher_norm), axis=-1)
    return loss


def calculate_attention_loss():
    #used when training single
    #layer_names = convOutputsDictTeacher.keys()

    loss_sum = None
    for layer_name in layer_names:
        if loss_sum is None:
            loss_sum = calculate_attention_loss_for_layer(layer_name)
        else:
            loss_sum = loss_sum + calculate_attention_loss_for_layer(layer_name)
    loss_sum = loss_sum / len(layer_names)

    return loss_sum

def format_layer_name(layer_name):
    # Remove counter (X) from resnete1X to enable matching between teacher and student
    layer_name = re.sub(r"resnete1\d", "resnete1", layer_name)
    return layer_name



class AttentionMatchingBinaryConvolution(BinaryConvolution):
    def __init__(self, channels, kernel_size=3, stride=1, padding=0, in_channels=0, dilation=1, bits=None, bits_a=None,
                 clip_threshold=None, activation_method=None, prefix=None, **kwargs):
        super(AttentionMatchingBinaryConvolution, self).__init__(channels, kernel_size=kernel_size, stride=stride, padding=padding, in_channels=in_channels, dilation=dilation,
                                                                 bits=bits, bits_a=bits_a, clip_threshold=clip_threshold, activation_method=activation_method, prefix=prefix, **kwargs)

        bin_layer_config = nn.binary_layer_config.get_values()
        self.is_student = bin_layer_config['attention_matching_student']
        if self.is_student:
            self.attention_map_dict = convOutputsDictStudent
        else:
            self.attention_map_dict = convOutputsDictTeacher
        layer_name = format_layer_name(self.name)
        layer_names.add(layer_name)

    def hybrid_forward(self, F, x):
        out = BinaryConvolution.hybrid_forward(self, F, x)
        self.add_to_outputs(out, self.name)
        return out

    def add_to_outputs(self, output, layer_name):
        # remove prefixes that differ between nets
        layer_name = format_layer_name(layer_name)
        self.attention_map_dict[layer_name] = output
