import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionUNet(torch.nn.Module):
    """
    UNet, down sampling & up sampling for global reasoning
    """

    def __init__(self, input_channels, class_number, **kwargs):
        super(AttentionUNet, self).__init__()

        down_channel = kwargs['down_channel'] # default = 256

        down_channel_2 = down_channel * 2 # down_channel_2 = 256 * 2 = 512
        up_channel_1 = down_channel_2 * 2 # up_channel_1 = 512 * 2 = 1024
        up_channel_2 = down_channel * 2 # up_channel_2 = 256 * 2 = 512

        self.inc = InConv(input_channels, down_channel)
        self.down1 = DownLayer(down_channel, down_channel_2)
        self.down2 = DownLayer(down_channel_2, down_channel_2)

        self.up1 = UpLayer(up_channel_1, up_channel_1 // 4)
        self.up2 = UpLayer(up_channel_2, up_channel_2 // 4)
        self.outc = OutConv(up_channel_2 // 4, class_number)

    def forward(self, attention_channels):
        """
        Given multi-channel attention map, return the logits of every one mapping into 3-class
        :param attention_channels:
        :return:
        """
        # attention_channels as the shape of: batch_size x channel x width x height
        x = attention_channels # torch.Size([1, 3, 120, 120])
        # print('x.size() = ', x.size())  # torch.Size([1, 3, 120, 120])

        x1 = self.inc(x)
        # print('x1.size() = ', x1.size())  # torch.Size([1, 256, 120, 120])

        x2 = self.down1(x1)
        # print('x2.size() = ', x2.size())  # torch.Size([1, 512, 60, 60])

        x3 = self.down2(x2)
        # print('x3.size() = ', x3.size())  # torch.Size([1, 512, 30, 30])

        x = self.up1(x3, x2)
        # print('1: x.size() = ', x.size())  # torch.Size([1, 256, 60, 60])

        x = self.up2(x, x1)
        # print('2: x.size() = ', x.size())  # torch.Size([1, 128, 120, 120])

        output = self.outc(x)
        # print('1: output.size() = ', output.size())  # torch.Size([1, 256, 120, 120])
        # attn_map as the shape of: batch_size x width x height x class
        output = output.permute(0, 2, 3, 1).contiguous()
        # print('2: output.size() = ', output.size())  # torch.Size([1, 120, 120, 256])
        return output


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.double_conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch) # in_ch = 3, out_ch = 256

    def forward(self, x):
        x = self.conv(x)
        return x


class DownLayer(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class UpLayer(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpLayer, self).__init__()
        if bilinear:
            # print('执行了 bilinear')
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        else:
            # print('没有执行执行了 bilinear')
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # print('x1.size() = ', x1.size())  # torch.Size([1, 512, 30, 30])
        x1 = self.up(x1)
        # print('up: x1.size() = ', x1.size())  # torch.Size([1, 512, 60, 60])
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # print('diffY = ', diffY) # 0
        # print('diffX = ', diffX) # 0
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
                        diffY // 2))
        # print('pad: x1.size() = ', x1.size())  # torch.Size([1, 512, 60, 60])
        x = torch.cat([x2, x1], dim=1)
        # print('1: x.size() = ', x.size())  # torch.Size([1, 1024, 60, 60])
        x = self.conv(x)
        # print('2: x.size() = ', x.size())  # torch.Size([1, 256, 60, 60])
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x