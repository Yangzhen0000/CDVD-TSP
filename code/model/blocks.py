import torch.nn as nn
import torch

###############################
# common
###############################

def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


###############################
# ResNet
###############################

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


###############################
# SpaceToDepth
###############################
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        space = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = space.size()
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        output = []
        for channel in range(s_depth):
            t_1 = space[:, :, :, channel].split(self.block_size, 2)
            stack = [t_t.contiguous().view(batch_size, d_height, self.block_size_sq) for t_t in t_1]
            channelO = torch.stack(stack, 1)
            output.append(channelO)
        output = torch.cat(output, 3)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output