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

    return VBDE_LAP(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_flow_net=load_flow_net, load_recons_net=load_recons_net,
                    flow_pretrain_fn=flow_pretrain_fn, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device)


class VBDE_LAP(nn.Module):
    '''
    based on CDVD_TSP
    remove: cascaded training and temporal sharpness prior
    add: laplacian mask
    '''
    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda'):
        super(VBDE_LAP, self).__init__()
        print("Creating VBDE_LAP Net")

        self.n_sequence = n_sequence
        self.device = device
        self.is_mask_filter = is_mask_filter

        assert n_sequence == 3, "Only support args.n_sequence=3; but get args.n_sequence={}".format(n_sequence)

        extra_channels = 1
        print('Concat laplacian mask')

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
        lap_grad = [utils.calc_LOG(im) for im in img_list_copy]

        # calculate the quantization mask
        mid_frame = img_list_copy[num_frames // 2]
        grad_sum = torch.zeros(mid_frame.size()[0], 1, mid_frame.size()[2], mid_frame.size()[3]).cuda()
        for i in range(num_frames):
            grad_sum = grad_sum + torch.abs(lap_grad[i])
        grad_mask = (grad_sum - torch.min(grad_sum))/(torch.max(grad_sum)-torch.min(grad_sum))

        # print(torch.max(grad_mask), torch.min(grad_mask))

        sum_mask = torch.ones_like(flow_mask_list[0])
        for i in range(num_frames):
            sum_mask = sum_mask * flow_mask_list[i]
        sum_mask = torch.sum(sum_mask, dim=1, keepdim=True)
        sum_mask = (sum_mask > 0).float()
        grad_mask = grad_mask * sum_mask

        # debugging
        # print("sum of the grad mask:{}".format(torch.sum(grad_mask)))
        return grad_mask

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