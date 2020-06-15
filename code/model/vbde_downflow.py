import torch
import torch.nn as nn
import torch.nn.functional as F
from model import recons_video
from model import flow_pwc
import time

def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    load_flow_net = True
    load_recons_net = False
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return VBDE_DOWNFLOW(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_flow_net=load_flow_net, load_recons_net=load_recons_net,
                    flow_pretrain_fn=flow_pretrain_fn, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device)


class VBDE_DOWNFLOW(nn.Module):
    '''
    based on CDVD_TSP
    remove: cascaded training and temporal sharpness prior
    add: none
    '''
    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda'):
        super(VBDE_DOWNFLOW, self).__init__()
        print("Creating VBDE_DOWNFLOW Net")

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence == 3, "Only support args.n_sequence=3; but get args.n_sequence={}".format(n_sequence)
        extra_channels = 0

        self.flow_net = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.recons_net = recons_video.RECONS_VIDEO(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]

        # Interation 1
        # restrict input sequences [0, 1, 2] to frame 1

        frame_list[2] = F.interpolate(frame_list[2], scale_factor=0.5, mode='bilinear')
        frame_list[0] = F.interpolate(frame_list[0], scale_factor=0.5, mode='bilinear')
        key_frame = F.interpolate(frame_list[1], scale_factor=0.5, mode='bilinear')

        warped01, _, _, flow_mask01 = self.flow_net(key_frame, frame_list[0])
        warped21, _, _, flow_mask21 = self.flow_net(key_frame, frame_list[2])
        warped01 = F.interpolate(warped01, scale_factor=2, mode='bilinear')
        warped21 = F.interpolate(warped21, scale_factor=2, mode='bilinear')

        concated = torch.cat([warped01, frame_list[1], warped21], dim=1)
        out, _ = self.recons_net(concated)

        return out