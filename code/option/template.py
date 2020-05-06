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
        
        args.test_every = 1
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
