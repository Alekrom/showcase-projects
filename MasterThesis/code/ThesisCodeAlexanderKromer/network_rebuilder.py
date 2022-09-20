import collections
from mxnet.gluon import nn, block
from mxnet import gluon
from models import resnet
from property_collector import PropertyCollector, LayerPropertyConv, LayerTypes
from models.resnet import ResNetV2Shared, BasicBlockV2, ResNetV2
from models.resnet_e import ResNetE1, BasicBlockE1, ResNetE1Shared
from attention_matching.attention_matching import AttentionMatchingBinaryConvolution

"""
    Build a Model with the architecture of the given supernet, that shares its latent weights, but can be quantized to a different bit width
"""

class NetworkRebuilder:
    def __init__(self, supernet, collector):
        self.supernet = supernet
        self.collector = collector
        self.stage_index = 0
        # start with two accounting for base nets created (for this shared net and for teacher)
        self.net_num = 2

    def rebuild_net(self, bits_w, bits_a):
        if type(self.supernet) == ResNetV2:
            subnet = ResNetV2Shared(self.supernet.channels, bits_w, bits_a, self.supernet.classes, prefix=f'resnete1{self.net_num}_')
        elif type(self.supernet == ResNetE1):
            subnet = ResNetE1Shared(self.supernet.channels, bits_w, bits_a, self.supernet.classes, prefix=f'resnete1{self.net_num}_')
        self.net_num += 1

        with subnet.name_scope():
            output_params = self.collector.get_output()
            subnet.output = self.build_dense(output_params)

            for (idx, feature) in enumerate(self.supernet.features):
                if isinstance(feature, nn.Conv2D):
                    layer_properties = self.collector.get_next_layer('initial', LayerTypes.Conv2d)
                    layer = self.build_2dConv_from_layer_properties(layer_properties)
                    subnet.features.add(layer)
                elif isinstance(feature, nn.BatchNorm):
                    subnet.features.add(nn.BatchNorm())
                elif isinstance(feature, nn.HybridSequential):
                    hybrSeq = nn.HybridSequential(prefix='stage%d_' % self.stage_index)
                    self.stage_index += 1
                    subnet.features.add(hybrSeq)
                    self.rebuild_stage(feature, hybrSeq)
                elif isinstance(feature, nn.Activation):
                    subnet.features.add(nn.Activation(feature._act_type))
                elif isinstance(feature, nn.GlobalAvgPool2D):
                    subnet.features.add(nn.GlobalAvgPool2D())
                elif isinstance(feature, nn.Flatten):
                    subnet.features.add(nn.Flatten())
                elif isinstance(feature, nn.MaxPool2D):
                    l_prop = self.collector.get_next_layer('initial', LayerTypes.maxPool2d)
                    maxPool = self.build_max_pool_2d(l_prop)
                    subnet.features.add(maxPool)
                else:
                    print("UNHANDLED FEATURE")

        self.collector.reset()
        return subnet


    def rebuild_stage(self, supernet_stage, subnet_stage):
        with subnet_stage.name_scope():
            for (key, value) in enumerate(supernet_stage._children):
                # Access element of ordered Dict, TODO refactor
                block = supernet_stage._children[value]
                if type(block) == resnet.BasicBlockV2:
                    subnet_block = self.copy_basic_block_V2(block)
                    subnet_stage.add(subnet_block)
                    self.rebuild_basic_block_v2_shared(block, subnet_block)
                elif type(block) == BasicBlockE1:
                    subnet_block = self.copy_basic_block_e1(block)
                    subnet_stage.add(subnet_block)
                    self.rebuild_basic_block_e1(block, subnet_block)
                elif (type(block)) == nn.Conv2D:
                    layer_properties = self.collector.get_next_layer(block.name, LayerTypes.Conv2d)
                    layer = self.build_2dConv_from_layer_properties(layer_properties)
                    subnet_stage.add(layer)
                elif isinstance(block, nn.Activation):
                    subnet_stage.add(nn.Activation(block._act_type))
                elif isinstance(block, nn.BatchNorm):
                    subnet_stage.add(nn.BatchNorm())
                else:
                    print("1")
                    print(type(block))
                    print("unexpected block")


    def rebuild_basic_block_v2_shared(self, supernet_block, subnet_block):
        # Handle downsample layer
        if supernet_block.downsample is not None:
            subnet_block.downsample = nn.HybridSequential(prefix='')
            ds_prop = self.collector.get_downsample(supernet_block.name)
            ds_layer = self.build_ActivatedQ(ds_prop)
            subnet_block.downsample.add(ds_layer)

        # Handle other layers
        for (key, layer) in enumerate(supernet_block.body):
            if type(layer) == nn.BinaryConvolution or type(layer) == AttentionMatchingBinaryConvolution or type(layer) == nn.ActivatedBinaryConvolution:
                layer_properties = self.collector.get_next_layer(supernet_block.name, LayerTypes.activatedConv)
                subnet_layer = self.build_ActivatedQ(layer_properties)
                subnet_block.body.add(subnet_layer)
            elif type(layer) == nn.BatchNorm:
                subnet_block.body.add(nn.BatchNorm())
            else:
                print("2")
                print(type(layer))
                print("unexpected block")

    def rebuild_basic_block_e1(self, supernet_block, subnet_block):
        # Handle downsample layer
        if supernet_block.downsample is not None:
            subnet_block.downsample = nn.HybridSequential(prefix='')
            if supernet_block.use_pooling:
                pool_prop = self.collector.get_downsample(supernet_block.name)
                pool_l = self.build_avg_pool_2d(pool_prop)
                subnet_block.downsample.add(pool_l)

            if supernet_block.use_fp:
                ds_prop = self.collector.get_downsample(supernet_block.name)
                ds_layer = self.build_2dConv_from_layer_properties(ds_prop)
                subnet_block.downsample.add(ds_layer)
            else:
                ds_prop = self.collector.get_downsample(supernet_block.name)
                ds_layer = self.build_ActivatedQ(ds_prop)
                subnet_block.downsample.add(ds_layer)
            subnet_block.downsample.add(nn.BatchNorm())

        # Handle other layers
        for (key, layer) in enumerate(supernet_block.body):
            if type(layer) == nn.BinaryConvolution or type(layer) == AttentionMatchingBinaryConvolution or type(layer) == nn.ActivatedBinaryConvolution:
                layer_properties = self.collector.get_next_layer(supernet_block.name, LayerTypes.activatedConv)
                subnet_layer = self.build_ActivatedQ(layer_properties)
                subnet_block.body.add(subnet_layer)
            elif type(layer) == nn.BatchNorm:
                subnet_block.body.add(nn.BatchNorm())
            else:
                print("3")
                print(type(layer))
                print("unexpected block")


    def build_2dConv_from_layer_properties(self, l_prop):
        return nn.Conv2D(channels=l_prop.channels, kernel_size=l_prop.kernel_size, strides=l_prop.strides,
                             padding=l_prop.padding, use_bias=l_prop.use_bias, groups=l_prop.groups,
                             params=l_prop.params, in_channels=l_prop.in_channels)

    def build_ActivatedQ(self, l_prop):
        # TODO double check the values set here
        subnet_layer = nn.activated_conv(channels=l_prop.channels, kernel_size=l_prop.kernel_size, stride=l_prop.strides,
                             padding=l_prop.padding, in_channels=l_prop.in_channels, qconv_params=l_prop.qconv_params)

        return subnet_layer

    def build_avg_pool_2d(self, l_prop):
        return nn.AvgPool2D(pool_size=l_prop.pool_size, strides=l_prop.strides, padding=l_prop.padding)

    def build_max_pool_2d(self, l_prop):
        return nn.MaxPool2D(pool_size=l_prop.pool_size, strides=l_prop.strides, padding=l_prop.padding)

    def copy_basic_block_V2(self, block):
        downsample = True
        if block.downsample is None:
            downsample = False
        copied_block = BasicBlockV2(block.channels, block.stride, downsample, block.in_channels, False)
        return copied_block

    def copy_basic_block_e1(self, block):
        copied_block = BasicBlockE1(block.channels, block.stride, block.downsample, in_channels=block.in_channels, init=False,
                                    use_fp=block.use_fp, use_pooling=block.use_pooling, write_on=block.write_on, slices=block.slices,
                                    num_groups=block.num_groups, prefix='')
        return copied_block

    def build_dense(self, l_prop):
        layer = nn.Dense(l_prop.classes, in_units=l_prop.in_units, params=l_prop.params)
        return layer
