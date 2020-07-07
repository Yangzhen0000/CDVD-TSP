import torch.nn as nn
import torch
import model.blocks as blocks

def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    return HybridC3D(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
        n_resblock=args.n_resblock, n_feat=args.n_feat, device=device)


class HybridC3D(nn.Module):
    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32, device='cuda'):
        super(HybridC3D, self).__init__()
        print("Creating HybridC3D Net")

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence == 3, "Only support args.n_sequence=3; but get args.n_sequence={}".format(n_sequence)

        InBlock = []
        # b, 3, 3, 256, 256
        InBlock.extend([nn.Sequential(
            nn.Conv3d(in_channels, n_feat, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )])
        # b, 32, 3, 256, 256
        InBlock.extend([blocks.ResBlock3D(n_feat, n_feat, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                        for _ in range(n_resblock)])
        # b, 32, 3, 256, 256
        # encoder1
        Encoder_first = [nn.Sequential(
            nn.Conv3d(n_feat, n_feat * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )]
        # b, 64, 3, 128, 128
        Encoder_first.extend([blocks.ResBlock3D(n_feat * 2, n_feat * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                              for _ in range(n_resblock)])
        # b, 64, 3, 128, 128
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv3d(n_feat * 2, n_feat * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )]
        # b, 128, 1, 64, 64
        Encoder_second.extend([blocks.ResBlock3D(n_feat * 4, n_feat * 4, kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), padding=(0, 1, 1)) for _ in range(n_resblock)])

        # decoder2
        Decoder_second = [nn.Sequential(
            nn.ConvTranspose3d(n_feat * 4, n_feat * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )]
        Decoder_second.extend([blocks.ResBlock3D(n_feat * 2, n_feat * 2, kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), padding=(0, 1, 1)) for _ in range(n_resblock)])

        # decoder1
        Decoder_first = [nn.Sequential(
            nn.ConvTranspose3d(n_feat * 2, n_feat, kernel_size=(1, 3, 3), stride=(1, 2, 2), 
                padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )]
        Decoder_first.extend([blocks.ResBlock3D(n_feat, n_feat, kernel_size=(1, 3, 3), 
            stride=(1, 1, 1), padding=(0, 1, 1)) for _ in range(n_resblock)])

        OutBlock = nn.Conv3d(n_feat, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = OutBlock


    def forward(self, x):
        reference = x[:, 1, :, :, :]
        in_sequence = x.permute(0, 2, 1, 3, 4)                              #N*C*D*H*W

        inblock = self.inBlock(in_sequence)                                 #N*n_feat*D*H*W
        encoder_first = self.encoder_first(inblock)                         #N*(2*n_feat)*D*(H/2)*(W/2)
        encoder_second = self.encoder_second(encoder_first)                 #N*(4*n_feat)*D*(H/4)*(W/4)
        decoder_second = self.decoder_second(encoder_second)
        decoder_first = self.decoder_first(decoder_second + encoder_first[:, :, 1:2, :, :])
        outBlock = self.outBlock(decoder_first + inblock[:, :, 1:2, :, :])
        out = torch.squeeze(outBlock)

        return out + reference
