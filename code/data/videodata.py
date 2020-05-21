import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import utils.utils as utils
import cv2
import random


class VIDEODATA(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.n_seq = args.n_sequence
        self.n_frames_per_video = args.n_frames_per_video
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []

        if train:
            self._set_filesystem(args.dir_data)
        else:
            # print("==========Loading validation data")
            self._set_filesystem(args.dir_data_test)

        self.images_gt, self.images_input = self._scan()

        self.num_video = len(self.images_gt)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if args.process:
            self.data_gt, self.data_input = self._load(self.images_gt, self.images_input)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'GT')
        self.dir_input = os.path.join(self.apath, 'INPUT')
        print("DataSet GT path:", self.dir_gt)
        print("DataSet INPUT path:", self.dir_input)


    def _scan(self):
        '''
        modify:
            self.n_frames_video: frame numbers of each video, [n1, n2, n3, ...]
        return:
            images_gt, images_input: [[frame names] for each video]
        '''
        vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))

        # for i in range(len(vid_input_names)):
        #     basename = os.path.basename(vid_input_names[i])
        #     dirname = os.path.dirname(vid_gt_names[i])
        #     name = os.path.join(dirname, basename)
        #     if name not in vid_gt_names:
        #         print("Missing file {}".format(name))

        # print("Length of gt_names:{}".format(len(vid_gt_names)))
        # print("Length of input_names:{}".format(len(vid_input_names)))
        assert len(vid_gt_names) == len(vid_input_names), "len(vid_gt_names) must equal len(vid_input_names)"

        # debugging setting: a small training set
        # if self.train:
        #      vid_gt_names = vid_gt_names[:1]
        #      vid_input_names = vid_input_names[:1]
        # if not self.train:
        #     print("=========={}".format(len(vid_gt_names)))
        #     print("=========={}".format(len(vid_input_names)))

        # random sample the dataset, comment for debugging
        if self.train:
            index_list = random.sample(range(0, len(vid_gt_names)), 200)
            sampled_gt_names = [vid_gt_names[index] for index in index_list]
            sample_input_names = [vid_input_names[index] for index in index_list]
        else:
            sampled_gt_names = vid_gt_names
            sample_input_names = vid_input_names

        images_gt = []
        images_input = []

        for vid_gt_name, vid_input_name in zip(sampled_gt_names, sample_input_names):
            if self.train:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))[:self.args.n_frames_per_video]
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))[:self.args.n_frames_per_video]
            else:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
            images_gt.append(gt_dir_names)
            images_input.append(input_dir_names)
            self.n_frames_video.append(len(gt_dir_names))
        return images_gt, images_input


    def _load(self, images_gt, images_input):
        data_input = []
        data_gt = []

        n_videos = len(images_gt)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            gts = np.array([cv2.imread(hr_name, cv2.IMREAD_UNCHANGED) for hr_name in images_gt[idx]])
            inputs = np.array([cv2.imread(lr_name, cv2.IMREAD_UNCHANGED) for lr_name in images_input[idx]])
            data_input.append(inputs)
            data_gt.append(gts)
        return data_gt, data_input

    def __getitem__(self, idx):
        if self.args.process:
            inputs, gts, filenames = self._load_file_from_loaded_data(idx)
        else:
            inputs, gts, filenames = self._load_file(idx)  # (nseq, h, w, c)

        # concat consecutive frames
        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq)]  # [(h, w, c), ...], len: nseq
        inputs_concat = np.concatenate(inputs_list, axis=2)  # (h, w, nseq*c)
        gts_list = [gts[i, :, :, :] for i in range(self.n_seq)]
        gts_concat = np.concatenate(gts_list, axis=2)

        inputs_concat, gts_concat = self.get_patch(inputs_concat, gts_concat, self.args.size_must_mode)  # (ps, ps, nseq*c)
        inputs_list = [inputs_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq)]  # [(ps, ps, c), ...], len: nseq
        gts_list = [gts_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq)]
        inputs = np.array(inputs_list)  # (nseq, ps, ps, c)
        gts = np.array(gts_list)  # (nseq, ps, ps, c)

        # convert numpy array to tensor with range (0, 1)
        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors) # [(c, ps, ps), ...], len: nseq, range: (0, 1)
        gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)

        # debugging output
        # print("==========> after normalizing, gt, min:{}, max:{}, lr, min:{}, max:{}".format(
        #     gt_tensors.min(), gt_tensors.max(), input_tensors.min(), input_tensors.max()))
        return torch.stack(input_tensors), torch.stack(gt_tensors), filenames  # (nseq, c, ps, ps), what's the use of filenames

    def __len__(self):
        if self.train:
            return self.num_frame * self.repeat
        else:
            return self.num_frame

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx

    # index the video number and frame number
    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        # gts = np.array([cv2.imread(hr_name, cv2.IMREAD_UNCHANGED) for hr_name in f_gts])  # shape, (nseq, h, w, c)
        # inputs = np.array([cv2.imread(lr_name, cv2.IMREAD_UNCHANGED) for lr_name in f_inputs])  # shape, (nseq, h, w, c)

        gts = []
        inputs = []
        for hr_name in f_gts:
            hr_img = cv2.imread(hr_name, cv2.IMREAD_UNCHANGED)
            while hr_img is None:
                print("Error in reading image {}".format(hr_name))
                hr_img = cv2.imread(hr_name, cv2.IMREAD_UNCHANGED)
            gts.append(hr_img)
            # debugging output
            # if os.path.basename(hr_name) == '003.png':
            #     cv2.imwrite("load_hr_003.png", hr_img)
        for lr_name in f_inputs:
            lr_img = cv2.imread(lr_name, cv2.IMREAD_UNCHANGED)
            while lr_img is None:
                print("Error in reading image {}".format(lr_name))
                lr_img = cv2.imread(lr_name, cv2.IMREAD_UNCHANGED)
            inputs.append(lr_img)
            # debugging output
            # if os.path.basename(lr_name) == '003.png':
            #     cv2.imwrite("load_lr_003.png", lr_img)
        gts = np.array(gts)
        inputs = np.array(inputs)

        # debug output
        # print("==========> gt.datatype={}, inputs.datatype={}".format(gts.dtype, inputs.dtype))
        # print("==========> after loading, gt, min:{}, max:{}, lr, min:{}, max:{}".format(
        #     gts.min(), gts.max(), inputs.min(), inputs.max()))

        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]

        return inputs, gts, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        gts = self.data_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        inputs = self.data_input[video_idx][frame_idx:frame_idx + self.n_seq]
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]]

        return inputs, gts, filenames

    def get_patch(self, input, gt, size_must_mode=1):
        if self.train:
            input, gt = utils.get_patch(input, gt, patch_size=self.args.patch_size)
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
            if not self.args.no_augment:
                input, gt = utils.data_augment(input, gt)
        else:
            h, w, c = input.shape

            # resize to accelerate the validation process
            input = cv2.resize(input, (960, 540))
            gt = cv2.resize(gt, (960, 540))

            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
        return input, gt
