import torch
import torch.nn as nn
import torch.nn.functional as F
from model import recons_video
from model import flow_pwc
from utils import utils

def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    load_flow_net = True
    load_recons_net = False
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True

    return VBDE_STEPMASK(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_flow_net=load_flow_net, load_recons_net=load_recons_net,
                    flow_pretrain_fn=flow_pretrain_fn, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device,
                   lbd=args.low_bitdepth, hbd=args.high_bitdepth)


class VBDE_STEPMASK(nn.Module):
    '''
    based on vbde
    remove: cascaded training and temporal sharpness prior
    add: quantization step mask
    '''
    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda', lbd=4, hbd=16):
        super(VBDE_STEPMASK, self).__init__()
        print("Creating VBDE_STEPMASK Net")

        self.n_sequence = n_sequence
        self.device = device
        self.is_mask_filter = is_mask_filter

        self.lbd = lbd
        self.hbd = hbd
        self.quantization_step = 2**(self.hbd - self.lbd)/(2**self.hbd - 1)

        # initialize quantization step filter
        top_kernel = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        down_kernel = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        left_kernel = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        right_kernel = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)

        self.top_filter = nn.Parameter(data=top_kernel, requires_grad=False)
        self.down_filter = nn.Parameter(data=down_kernel, requires_grad=False)
        self.left_filter = nn.Parameter(data=left_kernel, requires_grad=False)
        self.right_filter = nn.Parameter(data=right_kernel, requires_grad=False)

        assert n_sequence == 3, "Only support args.n_sequence=3; but get args.n_sequence={}".format(n_sequence)

        extra_channels = 1
        print('Concat quantization step mask')

        self.flow_net = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.recons_net = recons_video.RECONS_VIDEO(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def get_masks(self, img_list):
        num_frames = len(img_list)

        # detach backward
        img_list_copy = [img.detach() for img in img_list]
        # calculate the quantization mask
        mid_frame = img_list_copy[num_frames // 2]
        mid_frame = torch.mean(mid_frame, dim=1, keepdim=True)
        top_quant_map = torch.abs(F.conv2d(mid_frame, self.top_filter, bias=None, stride=1, padding=1))
        down_quant_map = torch.abs(F.conv2d(mid_frame, self.down_filter, bias=None, stride=1, padding=1))
        left_quant_map = torch.abs(F.conv2d(mid_frame, self.left_filter, bias=None, stride=1, padding=1))
        right_quant_map = torch.abs(F.conv2d(mid_frame, self.right_filter, bias=None, stride=1, padding=1))

        quant_mask = (top_quant_map - self.quantization_step < 10e-5).float()    \
                    + (down_quant_map - self.quantization_step < 10e-5).float()  \
                    + (left_quant_map - self.quantization_step < 10e-5).float()  \
                    + (right_quant_map - self.quantization_step < 10e-5).float()

        # debugging
        # print("sum of the quantization mask:{}".format(torch.sum(quantization_mask)))
        return quant_mask.cuda()

    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]

        # Interation 1
        # restrict input sequences [0, 1, 2] to frame 1
        warped01, _, _, flow_mask01 = self.flow_net(frame_list[1], frame_list[0])
        warped21, _, _, flow_mask21 = self.flow_net(frame_list[1], frame_list[2])

        frame_warp_list = [warped01, frame_list[1], warped21]
        luckiness = self.get_masks(frame_warp_list)


        concated = torch.cat([warped01, frame_list[1], warped21, luckiness], dim=1)
        out, _ = self.recons_net(concated)

        return out