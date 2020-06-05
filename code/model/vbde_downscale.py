import torch
import torch.nn as nn
from model import recons_video
from model import flow_pwc
from model.blocks import SpaceToDepth


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    load_flow_net = True
    load_recons_net = False
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return VBDE(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_flow_net=load_flow_net, load_recons_net=load_recons_net,
                    flow_pretrain_fn=flow_pretrain_fn, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device)


class VBDE(nn.Module):
    '''
    based on CDVD_TSP
    remove: cascaded training and temporal sharpness prior
    add: none
    '''
    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda'):
        super(VBDE, self).__init__()
        print("Creating VBDE Net")

        self.n_sequence = n_sequence
        self.device = device

        assert n_sequence == 3, "Only support args.n_sequence=3; but get args.n_sequence={}".format(n_sequence)

        extra_channels = 0
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

        self.space2depth = SpaceToDepth(2)
        self.depth2space = nn.PixelShuffle(2)

        self.flow_net = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device=device)
        self.recons_net = recons_video.RECONS_VIDEO(in_channels=in_channels, n_sequence=3, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat,
                                                    extra_channels=extra_channels)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        downsampled_frame_list = [self.space2depth(frame) for frame in frame_list]
        # Interation 1
        # restrict input sequences [0, 1, 2] to frame 1
        warped01, _, _, flow_mask01 = self.flow_net(frame_list[1], frame_list[0])
        warped21, _, _, flow_mask21 = self.flow_net(frame_list[1], frame_list[2])

        concated = torch.cat([warped01, frame_list[1], warped21], dim=1)
        out, _ = self.recons_net(concated)

        return out