import torch.nn as nn

class QCC(nn.Module):
    def __init__(self, method='floor', hbd=16, lbd=4, device='cuda'):
        super(QCC, self).__init__()
        self.method = method
        self.hbd = hbd
        self.lbd = lbd
        self.rgb_range = 2**hbd - 1
        self.L1_loss = nn.L1Loss()
        self.device = device

    def tensor2image(self, input_tensor):
        image = input_tensor.mul(self.rgb_range).clamp(0, self.rgb_range).round()
        return image

    def quantization(self, hbd_image):
        if self.method == 'floor':
            lbd_image = (hbd_image / (2**self.hbd-1)).floor() * (2**self.lbd-1)
        elif self.method == 'round':
            lbd_image = (hbd_image / (2**self.hbd-1)).round() * (2**self.lbd-1)
        return lbd_image

    def forward(self, x, y):
        x = self.tensor2image(x)
        y = self.tensor2image(y)

        lbd_x = self.quantization(x)
        lbd_y = self.quantization(y)

        qcc_loss = self.L1_loss(lbd_x, lbd_y)
        return qcc_loss
