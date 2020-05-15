import os
from data import videodata


class SDR4K(videodata.VIDEODATA):
    def __init__(self, args, name='SDR4k', train=True):
        super(SDR4K, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        if self.train:
            self.dir_gt = os.path.join(self.apath, 'SDR_10BIT_patch')
            self.dir_input = os.path.join(self.apath, 'SDR_4BIT_patch')
        else:
            self.dir_gt = os.path.join(self.apath, 'gt')
            self.dir_input = os.path.join(self.apath, 'input')
        print("DataSet gt path:", self.dir_gt)
        print("DataSet blur path:", self.dir_input)
