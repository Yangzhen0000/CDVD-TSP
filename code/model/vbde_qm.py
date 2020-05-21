import torch
import torch.nn as nn
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

    return VBDE_QM(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_flow_net=load_flow_net, load_recons_net=load_recons_net,
                    flow_pretrain_fn=flow_pretrain_fn, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device,
                   lbd=args.low_bitdepth, hbd=args.high_bitdepth)


class VBDE_QM(nn.Module):
    '''
    based on CDVD_TSP
    remove: cascaded training and temporal sharpness prior
    add: quantization mask
    '''
    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda', lbd=4, hbd=16):
        super(VBDE_QM, self).__init__()
        print("Creating VBDE_QM Net")

        self.n_sequence = n_sequence
        self.device = device
        self.is_mask_filter = is_mask_filter

        self.lbd = lbd
        self.hbd = hbd

        assert n_sequence == 3, "Only support args.n_sequence=3; but get args.n_sequence={}".format(n_sequence)

        extra_channels = 1
        print('Concat quantization mask')

        self.flow_net = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.recons_net = recons_video.RECONS_VIDEO(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def get_masks(self, img_list, flow_mask_list):
        num_frames = len(img_list)

        img_list_copy = [img.detach() for img in img_list]
        if self.is_mask_filter:
            img_list_copy = [utils.calc_meanFilter(im, n_channel=3, kernel_size=5) for im in img_list_copy]

        # calculate the quantization mask
        mid_frame = img_list_copy[num_frames // 2]
        diff = torch.zeros_like(mid_frame)
        for i in range(num_frames):
            diff = diff + (img_list_copy[i] - mid_frame)
        diff = torch.sum(diff, dim=1, keepdim=True)
        quantization_step = torch.tensor(2).pow(self.hbd - self.lbd)
        low_thres = (diff > 2.5*quantization_step).float()
        high_thres = (diff < 3.5*quantization_step).float()
        quantization_mask = (low_thres + high_thres) // 2

        sum_mask = torch.ones_like(flow_mask_list[0])
        for i in range(num_frames):
            sum_mask = sum_mask * flow_mask_list[i]
        sum_mask = torch.sum(sum_mask, dim=1, keepdim=True)
        sum_mask = (sum_mask > 0).float()
        quantization_mask = quantization_mask * sum_mask

        # debugging
        print("sum of the quantization mask:{}".format(torch.sum(quantization_mask)))
        return quantization_mask

    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]

        # Interation 1
        # restrict input sequences [0, 1, 2] to frame 1
        warped01, _, _, flow_mask01 = self.flow_net(frame_list[1], frame_list[0])
        warped21, _, _, flow_mask21 = self.flow_net(frame_list[1], frame_list[2])
        one_mask = torch.ones_like(flow_mask01)

        frame_warp_list = [warped01, frame_list[1], warped21]
        flow_mask_list = [flow_mask01, one_mask.detach(), flow_mask21]
        luckiness = self.get_masks(frame_warp_list, flow_mask_list)


        concated = torch.cat([warped01, frame_list[1], warped21, luckiness], dim=1)
        out, _ = self.recons_net(concated)

        return out