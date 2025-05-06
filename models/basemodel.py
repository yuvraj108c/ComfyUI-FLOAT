import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput


from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def print_architecture(self, verbose=False):
        name = type(self).__name__
        result = '-------------------%s---------------------\n' % name
        total_num_params = 0
        for i, (name, child) in enumerate(self.named_children()):
            if 'loss' in name:
                continue
            num_params = sum([p.numel() for p in child.parameters()])
            total_num_params += num_params
            if verbose:
                result += "%s: %3.3fM\n" % (name, (num_params / 1e6))
            for i, (name, grandchild) in enumerate(child.named_children()):
                num_params = sum([p.numel() for p in grandchild.parameters()])
                if verbose:
                    result += "\t%s: %3.3fM\n" % (name, (num_params / 1e6))
        result += '[Network %s] Total number of parameters : %.3f M\n' % (name, total_num_params / 1e6)
        result += '-----------------------------------------------\n'
        print(result)

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def get_parameters_for_train(self):
        return self.parameters()

    def forward(self):
        raise NotImplementedError()



# def linear_interpolation(features, seq_len):
#     features = features.transpose(1, 2)
#     output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
#     return output_features.transpose(1, 2)


# class Wav2Vec2Model(Wav2Vec2Model):
#     def __init__(self, config: Wav2Vec2Config):
#         super().__init__(config)

#     def forward(
#         self,
#         input_values,
#         seq_len,
#         attention_mask=None,
#         mask_time_indices=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         self.config.output_attentions = True

#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         extract_features = self.feature_extractor(input_values)
#         extract_features = extract_features.transpose(1, 2)
#         extract_features = linear_interpolation(extract_features, seq_len=seq_len)

#         if attention_mask is not None:
#             # compute reduced attention_mask corresponding to feature vectors
#             attention_mask = self._get_feature_vector_attention_mask(
#                 extract_features.shape[1], attention_mask, add_adapter=False
#             )

#         hidden_states, extract_features = self.feature_projection(extract_features)
#         hidden_states = self._mask_hidden_states(
#             hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
#         )

#         encoder_outputs = self.encoder(
#             hidden_states,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = encoder_outputs[0]

#         if self.adapter is not None:
#             hidden_states = self.adapter(hidden_states)

#         if not return_dict:
#             return (hidden_states, ) + encoder_outputs[1:]
#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )


#     def feature_extract(
#         self,
#         input_values,
#         seq_len,
#     ):
#         extract_features = self.feature_extractor(input_values)
#         extract_features = extract_features.transpose(1, 2)
#         extract_features = linear_interpolation(extract_features, seq_len=seq_len)

#         return extract_features

#     def encode(
#         self,
#         extract_features,
#         attention_mask=None,
#         mask_time_indices=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         self.config.output_attentions = True

#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if attention_mask is not None:
#             # compute reduced attention_mask corresponding to feature vectors
#             attention_mask = self._get_feature_vector_attention_mask(
#                 extract_features.shape[1], attention_mask, add_adapter=False
#             )
            

#         hidden_states, extract_features = self.feature_projection(extract_features)
#         hidden_states = self._mask_hidden_states(
#             hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
#         )

#         encoder_outputs = self.encoder(
#             hidden_states,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = encoder_outputs[0]

#         if self.adapter is not None:
#             hidden_states = self.adapter(hidden_states)

#         if not return_dict:
#             return (hidden_states, ) + encoder_outputs[1:]
#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )





def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer= nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[SEBasicBlock, SEBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SEBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, SEBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[SEBasicBlock, SEBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)




def se_resnet18(num_classes=1000, norm_layer=None):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=norm_layer)
    return model


def se_resnet34(num_classes=1000, norm_layer=None):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, norm_layer=norm_layer)
    return model


def se_resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


def se_resnet101(num_classes=1000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnet152(num_classes=1000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model