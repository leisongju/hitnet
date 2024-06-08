from __future__ import print_function

import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from einops import rearrange

import sys
from torchsummary import summary
from thop import profile


# basic convolution module
def conv3d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm3d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False)
    bn = nn.BatchNorm2d(out_planes)

    return nn.Sequential(conv, bn)


def tconvbn(in_planes, out_planes, kernel_size, stride):
    
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1,
                                            output_padding=1, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False)
    bn = nn.BatchNorm3d(out_planes)

    return nn.Sequential(conv, bn)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False, reluw=0.05, bn=True, relu=True):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(nout)
        if relu:
            self.prelu = nn.PReLU(init=reluw)
        self.use_bn = bn
        self.use_relu = relu

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        if self.use_bn:
            out = self.bn(out)
        if self.use_relu:
            out = self.prelu(out)
        return out
    
    
class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=1, deconv=False, is_3d=False, bn=True, relu=True, reluw = 0.05, bias=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.use_relu = relu
        self.use_bn = bn
        if relu:
            self.prelu = nn.PReLU(init=reluw)

        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=pad, stride=stride, bias=bias, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                      padding=pad,  stride=stride, bias=bias, **kwargs)
            if bn:
                self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=pad, stride=stride, bias=bias, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=pad, stride=stride, bias=bias, **kwargs)
            if bn:
                self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.prelu(x)
        return x


class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = Parameter(torch.ones(1,channel,1,1))
        self.bias = Parameter(torch.zeros(1,channel,1,1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias



class Encoder(nn.Module):
    
    def __init__(self, inchannel, infilter, outfilter, stride, pad_basic, reluw = 0.05):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=3, stride=stride, pad=pad_basic),  # 0
                                   depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=1)  # 1
                                   )

        self.conv2 = BasicBlock(infilter, outfilter, kernel_size=1, stride=1, pad=0)

        self.skip_connection = nn.Sequential(
            BasicBlock(inchannel, outfilter, kernel_size=1, stride=1, pad=pad_basic),
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=0)  # 0
        )

        self.prelu = nn.PReLU(init=reluw)

    def forward(self,x):

        x2 = self.conv1(x)

        x_skip = self.skip_connection(x)
        x2 = self.conv2(x2)

        x2 = x2 + x_skip

        x2 = self.prelu(x2)

        return x2


class Encoder2(nn.Module):

    def __init__(self, inchannel, outfilter, stride):
        super(Encoder2, self).__init__()

        self.conv1 = BasicBlock(inchannel, outfilter, kernel_size=7, stride=stride, pad=1)

        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=stride, padding=1)

    def forward(self, x):

        x_skip = self.maxpool(x)

        x = self.conv1(x)

        x = torch.cat((x, x_skip), dim=1)

        return x


class Decoder(nn.Module):

    def __init__(self, inchannel, infilter, pad_basic, pad_1, pad_2, pad_3, mode=None):
        super(Decoder, self).__init__()

        self.mode = mode

        if mode is None:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=4, stride=2, deconv=True, pad=pad_basic),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3)
                                       )
        else:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=3 + 2 * (pad_basic - 1), stride=1, pad=1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3)
                                       )

    def forward(self, x):

        if self.mode is not None:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)

        return self.conv1(x)


class Decoder2(nn.Module):

    def __init__(self, inchannel, infilter, outfilter, pad_basic, pad_1, pad_2, pad_3, mode=None):
        super(Decoder2, self).__init__()

        self.mode = mode

        if mode is None:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=4, stride=2, deconv=True, pad=pad_basic),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3),
                                       BasicBlock(infilter, outfilter, kernel_size=1, pad=1, bn=False, relu=False)
                                       )
        else:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=3 + 2 * (pad_basic - 1), stride=1, pad=1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3),
                                       BasicBlock(infilter, outfilter, kernel_size=1, pad=1, bn=False, relu=False)
                                       )

    def forward(self, x):

        if self.mode is not None:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)

        return self.conv1(x)



class Dpnet(nn.Module):
    def __init__(self):
        super(Dpnet, self).__init__()        
        input_channel = 1
        # mode : use bilinar upsampling instead of deconvolution
        mode = None
        
        # Encoder
        self.enc_layer1_1 = Encoder2(input_channel * 2, 8, 2)
        self.enc_layer1_2 = Encoder(8 + input_channel * 2, 11, 11, 1, 1)

        self.enc_layer2_1 = Encoder(11, 16, 32, 2, 0)
        self.enc_layer2_2 = Encoder(32, 16, 32, 1, 1)
        self.enc_layer2_3 = Encoder(32, 16, 32, 1, 1)

        self.enc_layer3_1 = Encoder(32, 16, 64, 2, 2)
        self.enc_layer3_2 = Encoder(64, 16, 64, 1, 1)
        self.enc_layer3_3 = Encoder(64, 16, 64, 1, 1)

        self.enc_layer4_1 = Encoder(64, 32, 128, 2, 1)
        self.enc_layer4_2 = Encoder(128, 32, 128, 1, 1)
        self.enc_layer4_3 = Encoder(128, 32, 128, 1, 1)

        self.enc_layer5_1 = Encoder(128, 32, 128, 2, 1)
        self.enc_layer5_2 = Encoder(128, 32, 128, 1, 1)
        self.enc_layer5_3 = Encoder(128, 32, 128, 1, 1)

        # Decoder
        self.dec_layer1 = Decoder(32, 16, 4, 1, 0, 1, mode=mode)
        self.dec_layer2 = Decoder(64, 16, 4, 0, 0, 0, mode=mode)
        self.dec_layer3 = Decoder(128, 16, 2, 0, 1, 0, mode=mode)
        self.dec_layer4 = Decoder(128, 32, 1, 1, 1, 1, mode=mode)
        
        # Skip Connection
        self.skip_layer1 = depthwise_separable_conv(11, 16, kernel_size=3, padding=3)
        self.skip_layer2 = depthwise_separable_conv(32, 16, kernel_size=3, padding=3)
        self.skip_layer3 = depthwise_separable_conv(64, 16, kernel_size=3, padding=3)
        self.skip_layer4 = depthwise_separable_conv(128, 32, kernel_size=3, padding=2)

        # Basic Blocks
        self.dec_layer1_b = BasicBlock(16, 32, kernel_size=1, pad=1, bn=False, relu=False)
        self.dec_layer2_b = BasicBlock(16, 32, kernel_size=1, pad=1, bn=False, relu=False)
        self.dec_layer3_b = BasicBlock(16, 64, kernel_size=1, pad=1, bn=False, relu=False)
        self.dec_layer4_b = BasicBlock(32, 128, kernel_size=1, pad=1, bn=False, relu=False)

        # Upsample layer (Last layer)
        self.last_layer = Decoder2(32, 8, 8, 4, 1, 0, 1, mode=mode)
        self.conv_last_layer5 = BasicBlock(128, 1, kernel_size=7, pad=1)
        self.conv_last_layer4 = BasicBlock(64, 1, kernel_size=7, pad=0)
        self.conv_last_layer3 = BasicBlock(32, 1, kernel_size=7, pad=1)
        self.conv_last_layer2 = BasicBlock(32, 1, kernel_size=7, pad=1)
        self.conv_last_layer1 = BasicBlock(8, 1, kernel_size=7, pad=1)
        
        # Activation function
        self.prelu = nn.PReLU(init=0.05)
        
        
        # Weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.init_weights()
                
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, l, r):
        x = torch.cat([l,r],dim=1)
        # Encoder layer 1
        x_layer1 = self.enc_layer1_1(x)
        x_layer1 = self.enc_layer1_2(x_layer1)

        # Encoder layer 2
        x_layer2 = self.enc_layer2_1(x_layer1)
        x_layer2 = self.enc_layer2_2(x_layer2)
        x_layer2 = self.enc_layer2_3(x_layer2)

        # Encoder layer 3
        x_layer3 = self.enc_layer3_1(x_layer2)
        x_layer3 = self.enc_layer3_2(x_layer3)
        x_layer3 = self.enc_layer3_3(x_layer3)

        # Encoder layer 4
        x_layer4 = self.enc_layer4_1(x_layer3)
        x_layer4 = self.enc_layer4_2(x_layer4)
        x_layer4 = self.enc_layer4_3(x_layer4)

        # Encoder layer 5
        x_layer5 = self.enc_layer5_1(x_layer4)
        x_layer5 = self.enc_layer5_2(x_layer5)
        x_layer5 = self.enc_layer5_3(x_layer5)

        # Decoder layer 5
        y_layer5 = self.dec_layer4(x_layer5)
        y_layer5 = self.prelu(y_layer5 + self.skip_layer4(x_layer4))  # torch.Size([1, 32, 50, 34])
        y_layer5 = self.dec_layer4_b(y_layer5)  # torch.Size([1, 128, 52, 36])

        # Decoder layer 4
        y_layer4 = self.dec_layer3(y_layer5)
        y_layer4 = self.prelu(y_layer4 + self.skip_layer3(x_layer3))  # torch.Size([1, 16, 100, 68])
        y_layer4 = self.dec_layer3_b(y_layer4)

        # Decoder layer 3
        y_layer3 = self.dec_layer2(y_layer4)
        y_layer3 = self.prelu(y_layer3 + self.skip_layer2(x_layer2))  # torch.Size([1, 16, 194, 130])
        y_layer3 = self.dec_layer2_b(y_layer3)

        # Decoder layer 2
        y_layer2 = self.dec_layer1(y_layer3)
        y_layer2 = self.prelu(y_layer2 + self.skip_layer1(x_layer1))  # torch.Size([1, 16, 386, 258])
        y_layer2 = self.dec_layer1_b(y_layer2)

        # Decoder layer 1
        y_layer1 = self.last_layer(y_layer2)
        
        if self.training:
            out5 = F.interpolate(self.conv_last_layer5(y_layer5), scale_factor=16,
                                            mode='bilinear', align_corners=True)
            out4 = F.interpolate(self.conv_last_layer4(y_layer4), scale_factor=8,
                                            mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.conv_last_layer3(y_layer3), scale_factor=4,
                                            mode='bilinear', align_corners=True)
            out2 = F.interpolate(self.conv_last_layer2(y_layer2), scale_factor=2,
                                            mode='bilinear', align_corners=True)
            out1 = self.conv_last_layer1(y_layer1)
            
            return [out1, out2, out3, out4, out5]        
        else:
            out1 = self.conv_last_layer1(y_layer1)
            return out1
    


    
if __name__ == '__main__':
    # 假设的option字典
    class Option:
        class Model:
            input_channel = 3
            mindisp = 0.1
            maxdisp = 1.0
        
        class Dataset:
            flip_lr = False
        
        class Loss:
            pass
        
        class Metric:
            pass
        
        model = Model()
        dataset = Dataset()
        loss = Loss()
        metric = Metric()
        batch_size = 1
        workers = 0
        pin_memory = False
        # 其他需要的参数可以在这里添加

    # 初始化模型
    option = Option()
    model = DPNET(option)

    # 创建一个随机的输入Tensor
    batch_size = 1
    channels = option.model.input_channel
    height = 256
    width = 256

    left_image = torch.randn(batch_size, channels * 2, height, width)
    right_image = torch.randn(batch_size, channels, height, width)


    device = 'cpu'


    input_tensor = torch.randn(batch_size, channels * 2, height, width).to(device)

    # 创建一个batch字典
    batch = left_image

    # 模型前向传播
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        outputs = model(batch)


    # 保存模型参数
    torch.save(model.state_dict(), 'dpnet_model.pth')
    print("模型参数已保存到 dpnet_model.pth")

    print(model(input_tensor).shape)
    # # 包装模型以适应torchsummary
    # class ModelSummaryWrapper(nn.Module):
    #     def __init__(self, model):
    #         super(ModelSummaryWrapper, self).__init__()
    #         self.model = model

    #     def forward(self, x):
    #         return self.model(x)

    # wrapped_model = ModelSummaryWrapper(model).to(device)

    # # 查看模型参数量
    # summary(wrapped_model, (channels * 2, height, width))

    # # 测量模型计算量
    # flops, params = profile(wrapped_model, inputs=(input_tensor,))
    # print(f"FLOPs: {flops}, Params: {params}")