def set_template(args):
    # task category
    args.task = 'VideoBDE'
    # network parameters
    args.n_feat = 32
    # loss
    args.loss = '1*L1+2*HEM'
    # learning rata strategy
    args.lr = 1e-4
    args.lr_decay = 100
    args.gamma = 0.1
    # data parameters
    args.data_train = 'SDR4K'
    args.data_test = 'SDR4K'
    args.n_sequence = 3
    args.n_frames_per_video = 100
    args.rgb_range = 65535
    args.size_must_mode = 4
    args.patch_size = 256
    args.dir_data = "/home/medialab/workspace/hdd/zhen/EDVR/datasets/SDR4k/train/" 
    args.dir_data_test = "/home/medialab/workspace/hdd/zhen/EDVR/datasets/SDR4k/val/"
    args.epochs = 500
    # test
    args.test_every = 1000
    args.print_every = 10
    if args.template == 'CDVD_TSP':
        args.model = "CDVD_TSP"
        args.n_sequence = 5
        args.n_resblock = 3
        args.lr_decay = 200
    elif args.template == 'VBDE':
        args.model = 'VBDE'
        args.n_resblock = 2
    elif args.template == 'VBDE_DOWNFLOW':
        args.model = 'VBDE_DOWNFLOW'
        args.n_resblock = 3
        args.lr_decay = 200
    elif args.template == 'VBDE_QM':
        args.model = 'VBDE_QM'
        args.n_resblock = 3
        args.lr_decay = 200
        # bit-depth parameters
        args.low_bitdepth = 4
        args.high_bitdepth = 16
    elif args.template == 'VBDE_LAP':
        args.model = 'VBDE_LAP'
        args.n_resblock = 3
        args.lr_decay = 50
    elif args.template == 'MOTION_NET':
        args.task = 'OpticalFlow'
        args.model = 'MOTION_NET'
        args.n_sequence = 2
        args.size_must_mode = 32
        args.loss = '1*MNL'
        # small learning rate for training optical flow
        args.lr = 1e-5
        args.lr_decay = 200
        args.gamma = 0.5
        args.data_train = 'SDR4K_FLOW'
        args.data_test = 'SDR4K_FLOW'
        args.video_samples = 500
    elif args.template == 'C3D':
        args.task = 'VideoBDE'
        args.model = 'C3D'
        args.n_resblock = 3
    elif args.template == 'HYBRID_C3D':
        args.model = 'HYBRID_C3D'
        args.n_resblock = 4
        args.scheduler = 'plateau'
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
