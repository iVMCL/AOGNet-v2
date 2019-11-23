# Modefied From: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# By Tianfu Wu
# Contact: tianfu_wu@ncsu.edu
import torch.nn as nn
from .aognet.operator_basic import FeatureNorm, MixtureBatchNorm2d, MixtureGroupNorm
from .config import cfg


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1,
                 norm_name=None, norm_groups=0, norm_k=0, norm_attention_mode=0):
        super(BasicBlock, self).__init__()
        if norm_name is None:
            norm_name = "BatchNorm2d"
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = FeatureNorm(norm_name, planes,
                               num_groups=norm_groups, num_k=norm_k,
                               attention_mode=norm_attention_mode)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = FeatureNorm(norm_name, planes,
                               num_groups=norm_groups, num_k=norm_k,
                               attention_mode=norm_attention_mode)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1,
                 norm_name=None, norm_groups=0, norm_k=0, norm_attention_mode=0,
                 norm_all_mix=False):
        super(Bottleneck, self).__init__()
        if norm_name is None:
            norm_name = "BatchNorm2d"
        if norm_all_mix:
            norm_name_base = norm_name
        else:
            if "BatchNorm2d" in norm_name:
                norm_name_base = "BatchNorm2d"
            elif "GroupNorm" in norm_name:
                norm_name_base = "GroupNorm"
            else:
                raise ValueError("Unknown norm.")
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = FeatureNorm(norm_name_base, width,
                               num_groups=norm_groups, num_k=norm_k,
                               attention_mode=norm_attention_mode)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = FeatureNorm(norm_name, width,
                               num_groups=norm_groups, num_k=norm_k,
                               attention_mode=norm_attention_mode)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = FeatureNorm(norm_name_base, planes * self.expansion,
                               num_groups=norm_groups, num_k=norm_k,
                               attention_mode=norm_attention_mode)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        replace_stride_with_dilation = cfg.resnet.replace_stride_with_dilation
        base_inplanes = cfg.resnet.base_inplanes
        norm_name = cfg.norm_name
        norm_groups = cfg.norm_groups
        norm_attention_mode = cfg.norm_attention_mode
        norm_all_mix = cfg.norm_all_mix
        replace_stride_with_avgpool = cfg.resnet.replace_stride_with_avgpool
        self.norm_name = norm_name
        self.norm_groups = norm_groups
        self.norm_ks = cfg.norm_k
        self.norm_attention_mode = norm_attention_mode
        self.norm_all_mix = norm_all_mix
        if norm_all_mix:
            self.norm_name_base = norm_name
        else:
            if "BatchNorm2d" in norm_name:
                self.norm_name_base = "BatchNorm2d"
            elif "GroupNorm" in norm_name:
                self.norm_name_base = "GroupNorm"
            else:
                raise ValueError("Unknown norm layer")
        if "Mixture" in norm_name:
            assert len(self.norm_ks) == len(layers) and any(self.norm_ks), \
                "Wrong mixture component specification (cfg.norm_k)"
        else:
            self.norm_ks = [0 for i in range(len(layers))]

        self.inplanes = base_inplanes
        self.extra_norm_ac = cfg.resnet.extra_norm_ac
        self.dilation = 1
        self.norm_k = self.norm_ks[0]
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if cfg.stem.imagenet_head7x7:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=cfg.stem.stem_kernel_size,
                                   stride=cfg.stem.stem_stride, padding=(
                                       cfg.stem.stem_kernel_size-1)//2,
                                   bias=False)
            self.bn1 = FeatureNorm(self.norm_name_base, self.inplanes,
                                   num_groups=norm_groups, num_k=self.norm_k,
                                   attention_mode=norm_attention_mode)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1) if cfg.dataset == 'imagenet' else None
        else:
            plane = self.inplanes // 2
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, plane, kernel_size=3,
                          stride=2, padding=1, bias=False),
                FeatureNorm(self.norm_name_base, plane,
                            num_groups=norm_groups, num_k=self.norm_k,
                            attention_mode=norm_attention_mode),
                nn.ReLU(inplace=True),
                nn.Conv2d(plane, plane, kernel_size=3,
                          stride=1, padding=1, bias=False),
                FeatureNorm(self.norm_name_base, plane,
                            num_groups=norm_groups, num_k=self.norm_k,
                            attention_mode=norm_attention_mode),
                nn.ReLU(inplace=True),
                nn.Conv2d(plane, self.inplanes, kernel_size=3,
                          stride=1, padding=1, bias=False)
            )
            self.bn1 = FeatureNorm(self.norm_name_base, self.inplanes,
                                   num_groups=norm_groups, num_k=self.norm_k,
                                   attention_mode=norm_attention_mode)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, base_inplanes, layers[0],
                                       replace_stride_with_avgpool=replace_stride_with_avgpool)
        self.norm_k = self.norm_ks[1]
        self.layer2 = self._make_layer(block, base_inplanes*2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       replace_stride_with_avgpool=replace_stride_with_avgpool)
        self.norm_k = self.norm_ks[2]
        self.layer3 = self._make_layer(block, base_inplanes*4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       replace_stride_with_avgpool=replace_stride_with_avgpool)
        self.layer4 = None
        outplanes = base_inplanes*4*block.expansion
        if len(layers) > 3:
            self.norm_k = self.norm_ks[3]
            self.layer4 = self._make_layer(block, base_inplanes*8, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2],
                                           replace_stride_with_avgpool=replace_stride_with_avgpool)
            outplanes = base_inplanes*8*block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outplanes, cfg.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, (MixtureBatchNorm2d, MixtureGroupNorm)):
                nn.init.normal_(m.weight_, 1, 0.1)
                nn.init.normal_(m.bias_, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if cfg.norm_zero_gamma_init:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    if isinstance(m.bn3, (MixtureBatchNorm2d, MixtureGroupNorm)):
                        nn.init.constant_(m.bn3.weight_, 0)
                        nn.init.constant_(m.bn3.bias_, 0)
                    else:
                        nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # TODO: handle mixture norm
                    nn.init.constant_(m.bn2.weight, 0)

    def _extra_norm_ac(self, out_channels, norm_k):
        return nn.Sequential(FeatureNorm(self.norm_name_base, out_channels,
                                         self.norm_groups, norm_k,
                                         self.norm_attention_mode),
                             nn.ReLU(inplace=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    replace_stride_with_avgpool=False):
        norm_name = self.norm_name
        norm_name_base = self.norm_name_base
        norm_groups = self.norm_groups
        norm_k = self.norm_k
        norm_attention_mode = self.norm_attention_mode
        norm_all_mix = self.norm_all_mix
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        downsample_op = []
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_stride = stride
            if replace_stride_with_avgpool and stride > 1:
                downsample_op.append(nn.AvgPool2d((stride, stride), stride))
                downsample_stride = 1

            downsample_op.append(
                conv1x1(self.inplanes, planes * block.expansion, downsample_stride))
            downsample_op.append(FeatureNorm(norm_name_base, planes * block.expansion,
                                             num_groups=norm_groups, num_k=norm_k,
                                             attention_mode=norm_attention_mode))

        if len(downsample_op) > 0:
            downsample = nn.Sequential(*downsample_op)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_name=norm_name, norm_groups=norm_groups, norm_k=norm_k,
                            norm_attention_mode=norm_attention_mode,
                            norm_all_mix=norm_all_mix))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_name=norm_name, norm_groups=norm_groups, norm_k=norm_k,
                                norm_attention_mode=norm_attention_mode,
                                norm_all_mix=norm_all_mix))

        if self.extra_norm_ac:
            layers.append(self._extra_norm_ac(self.inplanes, norm_k))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained=False, progress=True, **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained=False, progress=True, **kwargs)


def resnext101_64x4d(**kwargs):
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet('resnext101_64x4d', Bottleneck, [3, 4, 23, 3],
                   pretrained=False, progress=True, **kwargs)
