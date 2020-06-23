import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import utils.utils as utils
import cv2
import random


class SDR4K_FLOW(data.Dataset):
    def __init__(self, args, name='SDR4k', train=True):
        super(SDR4K_FLOW, self).__init__()
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
            self._set_filesystem(args.dir_data_test)

        self.images = self._scan()
        self.num_video = len(self.images)
        self.num_frame = sum(self.n_frames_video) - len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        if self.train:
            self.dir = os.path.join(self.apath, 'SDR_4BIT_patch')
        else:
            self.dir = os.path.join(self.apath, 'input')
        print("DataSet path:", self.dir)

    def _scan(self):
        vid_names = sorted(glob.glob(os.path.join(self.dir, '*')))
        if self.train:
            index_list = random.sample(range(0, len(vid_names)), 200)
            sampled_names = [vid_names[index] for index in index_list]
        else:
            sampled_names = vid_names

        images = []

        for vid_name in sampled_names:
            if self.train:
                images_names = sorted(glob.glob(os.path.join(vid_name, '*')))[:self.args.n_frames_per_video]
            else:
                images_names = sorted(glob.glob(os.path.join(vid_name, '*')))
            images.append(images_names)
            self.n_frames_video.append(len(images_names))
        return images

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def __getitem__(self, idx):
        n_poss_frames = [n - 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        
        frame1_name = self.images[video_idx][frame_idx]
        frame2_name = self.images[video_idx][frame_idx+1]
        frame1 = cv2.imread(frame1_name, cv2.IMREAD_UNCHANGED)
        while frame1 is None:
            print("Error in reading image {}".format(frame1_name))
            frame1 = cv2.imread(frame1_name, cv2.IMREAD_UNCHANGED)
        frame2 = cv2.imread(frame2_name, cv2.IMREAD_UNCHANGED)
        while frame2 is None:
            print("Error in reading image {}".format(frame2_name))
            frame2 = cv2.imread(frame2_name, cv2.IMREAD_UNCHANGED)
        filename = []
        filename.append(os.path.split(os.path.dirname(frame1_name))[-1] + '.' + \
                                 os.path.splitext(os.path.basename(frame1_name))[0])
        filename.append(os.path.split(os.path.dirname(frame2_name))[-1] + '.' + \
                                 os.path.splitext(os.path.basename(frame2_name))[0])

        frame1_patch, frame2_patch = self.get_patch(frame1, frame2, self.args.size_must_mode)

        frame1_patch = np.ascontiguousarray(frame1_patch.astype('float64').transpose((2, 0, 1)))
        frame1_tensor = torch.from_numpy(frame1_patch).float()
        frame1_tensor.mul_(1 / self.args.rgb_range)
        
        frame2_patch = np.ascontiguousarray(frame2_patch.astype('float64').transpose((2, 0, 1)))
        frame2_tensor = torch.from_numpy(frame2_patch).float()
        frame2_tensor.mul_(1 / self.args.rgb_range)

        return frame1_tensor, frame2_tensor, filename

    def __len__(self):
        return self.num_frame

    def get_patch(self, img1, img2, size_must_mode=1):
        if self.train:
            img1, img2 = utils.get_patch(img1, img2, patch_size=self.args.patch_size)
            h, w, c = img1.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            img1, img2 = img1[:new_h, :new_w, :], img2[:new_h, :new_w, :]
            if not self.args.no_augment:
                img1, img2 = utils.data_augment(img1, img2)
        else:
            # resize to accelerate the validation process
            img1 = cv2.resize(img1, (960, 540))
            img2 = cv2.resize(img2, (960, 540))

            h, w, c = img1.shape

            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            img1, img2 = img1[:new_h, :new_w, :], img2[:new_h, :new_w, :]
        return img1, img2
