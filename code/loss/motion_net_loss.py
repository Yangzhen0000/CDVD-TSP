import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from torch.autograd import Variable


class MNL(nn.Module):
    """docstring for MNL"""
    def __init__(self, device='cuda'):
        super(MNL, self).__init__()

        self.flow_scale_factor = 0.625

        self.ssim_loss_weight = 0.16
        self.ssim_kernel_size = 8
        self.ssim_stride = 8
        self.ssim_channel = 3
        self.ssim_val_range = 1
        
        self.cal_ssim_loss = SSIM_Loss(self.ssim_kernel_size, self.ssim_stride, self.ssim_val_range, self.ssim_channel)
        self.cal_photometric_loss = L1_Charbonnier_Loss()
        self.cal_smooth_loss = TVLoss()

        self.ssim_loss_weight = 1
        self.photometric_loss_weight = 1
        self.smooth_loss_weight = 10

        self.device = device

    def warp(self, img, flow):
        B, C, H, W = img.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(self.device)
        vgrid = Variable(grid) + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(img, vgrid, padding_mode='border')
        mask = torch.autograd.Variable(torch.ones(img.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output, mask

    def forward(self, img1, img2, flow):
        # calculate photometric loss
        batch_size, _, flow_h, flow_w = flow.size()
        downsampled_img1 = F.interpolate(img1, [flow_h, flow_w])
        downsampled_img2 = F.interpolate(img2, [flow_h, flow_w])
        scaled_flow = flow * self.flow_scale_factor
        warped_img2, _ = self.warp(downsampled_img2, scaled_flow)
        photometric_loss = self.cal_photometric_loss(downsampled_img1, warped_img2)

        # calculate ssim loss
        ssim_loss = self.cal_ssim_loss(downsampled_img1, warped_img2)

        # calculate smoothness loss on X axis
        x_smoothness_loss = self.cal_smooth_loss(flow[:, :1, :, :])
        y_smoothness_loss = self.cal_smooth_loss(flow[:, 1:, :, :])
        
        # print(photometric_loss, ssim_loss, x_smoothness_loss, y_smoothness_loss)
        return (self.photometric_loss_weight*photometric_loss + self.ssim_loss_weight*ssim_loss + \
                self.smooth_loss_weight*(x_smoothness_loss+y_smoothness_loss)) / batch_size


class L1_Charbonnier_Loss(nn.Module):
    """docstring for L1_Charbonnier_loss"""
    def __init__(self):
        super(L1_Charbonnier_Loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        _, _, h, w = x.size()
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error) / (h * w)
        return loss


class SSIM_Loss(torch.nn.Module):
    def __init__(self, window_size=8, stride=8, val_range=1, channel=3, size_average=True):
        super(SSIM_Loss, self).__init__()

        self.device = 'cuda'
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = channel
        self.window = self.create_window(self.window_size, self.channel).to(self.device)
        self.stride = stride
 
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
     
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, stride, window, size_average, val_range, full=False):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1
     
            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range
     
        padd = 0
        _, channel, height, width = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel=channel).to(img1.device)
     
        mu1 = F.conv2d(img1, window, stride=stride, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, stride=stride, padding=padd, groups=channel)
     
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
     
        sigma1_sq = F.conv2d(img1 * img1, window, stride=stride, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, stride=stride, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, stride=stride, padding=padd, groups=channel) - mu1_mu2
     
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
     
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity
     
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
     
        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
     
        if full:
            return ret, cs
        return ret

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return self.ssim(img1, img2, self.stride, self.window, self.size_average, self.val_range)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)


if __name__ == '__main__':
    img1 = Variable(torch.rand(1, 3, 256, 256).cuda(), requires_grad=True)
    img2 = Variable(torch.rand(1, 3, 256, 256).cuda(), requires_grad=True)
    flow = Variable(torch.rand(1, 2, 16, 16).cuda(), requires_grad=True)
    loss = MNL()
    l = loss(img1, img2, flow)
    l.backward()
    print(l)