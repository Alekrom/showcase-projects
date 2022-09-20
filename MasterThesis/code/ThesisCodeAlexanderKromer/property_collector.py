from enum import Enum

"""
    Methods to collect the parameters of the layers of a given network
    Used for weight sharing between models in the ensemble
"""

class LayerTypes(Enum):
    Conv2d = 'conv2d'
    activatedConv = 'activatedConv'
    dense = 'dense'
    avgPool = 'avgPool'
    maxPool2d = 'maxPool2d'
    activation = 'activation'

class LayerProperty:

    def __init__(self, params, layer_type):
        self.params = params
        self.layer_type = layer_type


class LayerPropertyConv(LayerProperty):
    # Defaults are the same as in nn.Conv2D
    def __init__(self, channels, kernel_size, params, layer_type, strides=(1,1), padding=(0,0), groups=1, use_bias=True, in_channels=0, qconv_params=None):
        super(LayerPropertyConv, self).__init__(params, layer_type)
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.groups = groups
        self.qconv_params = qconv_params
        self.in_channels = in_channels


class LayerPropertyDense(LayerProperty):
    # Defaults are the same as in nn.Dense
    def __init__(self,  classes, params, layer_type, in_units=0):
        super(LayerPropertyDense, self).__init__(params, layer_type)
        self.classes = classes
        self.in_units = in_units


class LayerPropertyAvgPooling(LayerProperty):
    def __init__(self, params, layer_type, pool_size, strides, padding):
        super(LayerPropertyAvgPooling, self).__init__(params, layer_type)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding


class LayerPropertyMaxPool2d(LayerProperty):
    def __init__(self, params, layer_type, pool_size, strides, padding):
        super(LayerPropertyMaxPool2d, self).__init__(params, layer_type)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding


class LayerPropertyActivation(LayerProperty):
    def __init__(self, layer_type, activation_string):
        super(LayerPropertyActivation, self).__init__(None, layer_type)
        self.activation_string = activation_string


class PropertyCollector:

    def __init__(self):
        self.properties = {}
        self.next_layer = {}
        self.downsample = {}
        self.next_downsample = {}
        self.output = None

    def add_layer_properties(self, layer_params, identifier):
        if identifier not in self.properties:
            self.properties[identifier] = []
            self.next_layer[identifier] = 0
        self.properties[identifier].append(layer_params)

    def add_downsample(self, properties, identifier):
        if identifier not in self.downsample:
            self.downsample[identifier] = []
            self.next_downsample[identifier] = 0

        self.downsample[identifier].append(properties)

    def add_output(self, properties):
        self.output = properties

    def get_output(self):
        return self.output

    def get_next_layer(self, identifier, layer_type):
        layer_properties = self.properties[identifier][self.next_layer[identifier]]
        self.next_layer[identifier] += 1
        if layer_properties.layer_type != layer_type:
            print("layer types do not match")
            exit("error")
        return layer_properties

    def get_downsample(self, identifier):
        layer = self.downsample[identifier][self.next_downsample[identifier]]
        self.next_downsample[identifier] += 1
        return layer

    def reset(self):
        for key in self.next_layer.keys():
            self.next_layer[key] = 0

        for key in self.next_downsample.keys():
            self.next_downsample[key] = 0