import torch
import torch.nn as nn

def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    return MotionNet()


class MotionNet(nn.Module):
    """docstring for MotionNet"""
    def __init__(self):
        super(MotionNet, self).__init__()
        print("Creating MotionNet")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.flow6 = nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.upsample_flow6to5 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.xconv5 = nn.Sequential(
            nn.Conv2d(in_channels=512+2+512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.flow5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.upsample_flow5to4 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.xconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256+2+512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.flow4 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.upsample_flow4to3 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.xconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128+2+256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.flow3 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.upsample_flow3to2 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.xconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64+2+128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.flow2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.flow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)


    def forward(self, tensor1, tensor2):
        frames = torch.cat([tensor1, tensor2], dim=1)
        conv1_feat = self.conv1(frames)
        conv1_1_feat = self.conv1_1(conv1_feat)
        conv2_feat = self.conv2(conv1_1_feat)
        conv2_1_feat = self.conv2_1(conv2_feat)
        conv3_feat = self.conv3(conv2_1_feat)
        conv3_1_feat = self.conv3_1(conv3_feat)
        conv4_feat = self.conv4(conv3_1_feat)
        conv4_1_feat = self.conv4_1(conv4_feat)
        conv5_feat = self.conv5(conv4_1_feat)
        conv5_1_feat = self.conv5_1(conv5_feat)
        conv6_feat = self.conv6(conv5_1_feat)
        conv6_1_feat = self.conv6_1(conv6_feat)

        flow6_feat = self.flow6(conv6_1_feat)
        upsample_flow6to5_feat = self.upsample_flow6to5(flow6_feat)
        deconv5_feat = self.deconv5(conv6_1_feat)
        concat_5_feat = torch.cat([deconv5_feat, upsample_flow6to5_feat, conv5_1_feat], dim=1)
        xconv5_feat = self.xconv5(concat_5_feat)

        flow5_feat = self.flow5(xconv5_feat)
        upsample_flow5to4_feat = self.upsample_flow5to4(flow5_feat)
        deconv4_feat = self.deconv4(xconv5_feat)
        concat_4_feat = torch.cat([deconv4_feat, upsample_flow5to4_feat, conv4_1_feat], dim=1)
        xconv4_feat = self.xconv4(concat_4_feat)

        flow4_feat = self.flow4(xconv4_feat)
        upsample_flow4to3_feat = self.upsample_flow4to3(flow4_feat)
        deconv3_feat = self.deconv3(xconv4_feat)
        concat_3_feat = torch.cat([deconv3_feat, upsample_flow4to3_feat, conv3_1_feat], dim=1)
        xconv3_feat = self.xconv3(concat_3_feat)

        flow3_feat = self.flow3(xconv3_feat)
        upsample_flow3to2_feat = self.upsample_flow3to2(flow3_feat)
        deconv2_feat = self.deconv2(xconv3_feat)
        concat_2_feat = torch.cat([deconv2_feat, upsample_flow3to2_feat, conv2_1_feat], dim=1)
        xconv2_feat = self.xconv2(concat_2_feat)

        flow2_feat = self.flow2(xconv2_feat)
        flow_out = self.flow(flow2_feat)

        return flow_out

