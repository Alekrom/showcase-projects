class BinaryLayerConfigParams:
    def __init__(self, grad_cancel=1.0, bits=1, bits_a=1, activation="det_sign",
                 weight_quantization="det_sign", approximation="", post_block_activation="", attention_matching_student=None):
        self.grad_cancel = grad_cancel
        self.bits = bits
        self.bits_a = bits_a
        self.activation = activation
        self.weight_quantization = weight_quantization
        self.approximation = approximation
        self.post_block_activation = post_block_activation
        self.attention_matching_student = attention_matching_student

    def describe(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())


def add_properties_if_collector(properties, collector, identifier, downsample=False):
    if collector is not None:
        if downsample:
            collector.add_downsample(properties, identifier)
        else:
            collector.add_layer_properties(properties, identifier)

def output_list_to_dict(net, output):
    outputs_dict = {}
    for (idx, output) in enumerate(output):
        outputs_dict[net.nets[idx].bits] = output


    return outputs_dict