def set_template(args):
    if args.template == 'CDVD_TSP':
        args.task = "VideoBDE"
        args.model = "CDVD_TSP"
        args.n_sequence = 5
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.data_train = 'SDR4K'
        args.data_test = 'SDR4K'
        args.rgb_range = 65535
        args.test_every = 1000
    if args.template == 'VBDE':
        args.task = 'VideoBDE'
        args.model = 'VBDE'
        args.n_sequence = 3
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.data_train = 'SDR4K'
        args.data_test = 'SDR4K'
        args.rgb_range = 65535
        args.test_every = 1000
        args.print_every = 10

        # update learning rate by gamma times every lr_decay epochs
        args.lr_decay = 100
        args.gamma = 0.1

    if args.template == 'VBDE_QM':
        args.task = 'VideoBDE'
        args.model = 'VBDE_QM'
        args.n_sequence = 3
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.patch_size = 256
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 200
        args.data_train = 'SDR4K'
        args.data_test = 'SDR4K'
        args.rgb_range = 65535
        args.test_every = 1000
        args.print_every = 10

        # update learning rate by gamma times every lr_decay epochs
        args.lr_decay = 100
        args.gamma = 0.1

        # bit-depth parameters
        args.low_bitdepth = 4
        args.high_bitdepth = 16

    if args.template == 'VBDE_LAP':
        args.task = 'VideoBDE'
        args.model = 'VBDE_LAP'
        args.n_sequence = 3
        args.n_frames_per_video = 100
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.patch_size = 256
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.data_train = 'SDR4K'
        args.data_test = 'SDR4K'
        args.rgb_range = 65535
        args.test_every = 1000
        args.print_every = 10

        # update learning rate by gamma times every lr_decay epochs
        args.lr_decay = 50
        args.gamma = 0.1
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
