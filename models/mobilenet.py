# from https://raw.githubusercontent.com/pytorch/vision/master/torchvision/models/mobilenet.py
# Modified by Tianfu Wu
# Contact: tianfu_wu@ncsu.edu

from torch import nn
from .aognet.operator_basic import FeatureNorm, MixtureBatchNorm2d, MixtureGroupNorm
from .config import cfg


__all__ = ['MobileNetV2', 'mobilenet_v2']


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                norm_name=None, norm_groups=0, norm_k=0, norm_attention_mode=0):
        if norm_name is None:
            norm_name = "BatchNorm2d"
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            #nn.BatchNorm2d(out_planes),
            FeatureNorm(norm_name, out_planes,
                        num_groups=norm_groups, num_k=norm_k,
                        attention_mode=norm_attention_mode),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,
                norm_name=None, norm_groups=0, norm_k=0, norm_attention_mode=0):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        if norm_name is None:
            norm_name = "BatchNorm2d"
        if "BatchNorm2d" in norm_name:
            norm_name_base = "BatchNorm2d"
        elif "GroupNorm" in norm_name:
            norm_name_base = "GroupNorm"
        else:
            raise ValueError("Unknown norm.")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1,
                        norm_name=norm_name_base, norm_groups=norm_groups,
                        norm_k=norm_k, norm_attention_mode=norm_attention_mode))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
                        norm_name=norm_name, norm_groups=norm_groups,
                        norm_k=norm_k, norm_attention_mode=norm_attention_mode),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            #nn.BatchNorm2d(oup),
            FeatureNorm(norm_name_base, oup,
                        num_groups=norm_groups, num_k=norm_k,
                        attention_mode=norm_attention_mode),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        norm_name = cfg.norm_name
        norm_groups = cfg.norm_groups
        norm_k = cfg.norm_k
        norm_attention_mode = cfg.norm_attention_mode
        if norm_name is None:
            norm_name = "BatchNorm2d"
        if "BatchNorm2d" in norm_name:
            norm_name_base = "BatchNorm2d"
        elif "GroupNorm" in norm_name:
            norm_name_base = "GroupNorm"
        else:
            raise ValueError("Unknown norm.")

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2,
                        norm_name=norm_name_base, norm_groups=norm_groups,
                        norm_k=-1, norm_attention_mode=norm_attention_mode)]
        # building inverted residual blocks
        for j, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t,
                                        norm_name=norm_name, norm_groups=norm_groups,
                                        norm_k=norm_k[j], norm_attention_mode=norm_attention_mode))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1,
                                    norm_name=norm_name_base, norm_groups=norm_groups,
                                    norm_k=-1, norm_attention_mode=norm_attention_mode))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (MixtureBatchNorm2d, MixtureGroupNorm)):
                nn.init.normal_(m.weight_, 1, 0.1)
                nn.init.normal_(m.bias_, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
