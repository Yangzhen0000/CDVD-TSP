import os
import time
import torch
import numpy as np
import cv2
import argparse
from model.vbde_stepmask import VBDE_STEPMASK
from tqdm import tqdm


##########################################
# Open a log file                       ##
# Write log message to file and stdout  ##
##########################################
class TraverseLogger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


##########################################
# Read in 8-bit videos                  ##
# Write out enhanced 10-bit videos      ##
##########################################
class VideoEnhancer:
    def __init__(self, args):
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        self.n_seq = args.n_seq
        self.size_must_mode = args.size_must_mode
        self.in_rgb_range = args.in_rgb_range
        self.out_rgb_range = args.out_rgb_range
        self.device = 'cuda'

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = TraverseLogger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = VBDE_STEPMASK(
            in_channels=3, n_sequence=self.n_seq, out_channels=3, n_resblock=3, n_feat=32,
            is_mask_filter=True, device=self.device
        )
        self.net.load_state_dict(torch.load(self.model_path), strict=True)
        self.net = self.net.to(self.device)
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()

    def infer(self):
        with torch.no_grad():
            total_forward_time = {}
            videos = sorted(os.listdir(self.data_path))
            for v in tqdm(videos):
                video_forward_time = []
                video_cap = cv2.VideoCapture(v)
                pure_name = os.path.basename(v)[:-4]
                frame_size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Reading video {:s} with {:d} frames of size ({:d}, {:d})".format(v, frame_num,
                                                                                        frame_size[0], frame_size[1]))
                fps = float(video_cap.get(cv2.CAP_PROP_FPS))
                video_out = os.path.join(self.result_path, pure_name+'_out.mp4')
                # ???Is VideoWriter support writing 10bit video???
                # FOURCC http://www.fourcc.org/codecs.php
                video_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('H', 'E', 'V', 'C'), fps)
                is_start = True
                while video_cap.get(cv2.CAP_PROP_POS_FRAMES) <= frame_num:
                    success, current_frame = video_cap.read()
                    if is_start:
                        previous_frame = current_frame
                        reference_frame = current_frame
                        _, next_frame = video_cap.read()
                    elif not success:
                        previous_frame = reference_frame
                        reference_frame = next_frame
                    else:
                        previous_frame = reference_frame
                        reference_frame = next_frame
                        next_frame = current_frame
                    input_seqs = [previous_frame, reference_frame, next_frame]
                    start_time = time.time()

                    # resize to avoid CUDA OUT OF MEMORY
                    resized_inputs = [cv2.resize(p, (960, 540)) for p in input_seqs]
                    h, w, c = resized_inputs[self.n_seq // 2].shape
                    new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                    cropped_inputs = [im[:new_h, :new_w, :] for im in resized_inputs]

                    in_tensor = self.numpy2tensor(cropped_inputs, rgb_range=self.in_rgb_range).to(self.device)
                    preprocess_time = time.time()
                    output = self.net(in_tensor)
                    forward_time = time.time()
                    output_img = self.tensor2numpy(output, rgb_range=self.out_rgb_range)

                    video_writer.write(output_img)

                    video_forward_time.append(forward_time)
                    total_forward_time[v] = video_forward_time

                    postprocess_time = time.time()

                    self.logger.write_log(
                        '> {}, pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, preprocess_time - start_time, forward_time - preprocess_time,
                                    postprocess_time - forward_time, postprocess_time - start_time))
                cv2.destroyAllWindows()
                video_cap.release()
                video_writer.release()

            sum_forward = 0.
            n_img = 0
            for k in total_forward_time.keys():
                self.logger.write_log("# Video:{}, AVG-FORWARD-TIME={:.4}".format(
                    k, sum(total_forward_time[k]) / len(total_forward_time[k])))
                sum_forward += sum(total_forward_time[k])
                n_img += len(total_forward_time[k])
            self.logger.write_log("# Total AVG-FORWARD-TIME={:.4}".format(sum_forward / n_img))

    def numpy2tensor(self, input_seq, rgb_range=255):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(1 / rgb_range)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = torch.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=255):
        img = tensor.mul(rgb_range).clamp(0, rgb_range).round()
        img = img[0].data
        if rgb_range == 255:
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        elif rgb_range == 65535:
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint16)
        else:
            raise ValueError('Invalid RGB range {:d}'.format(rgb_range))
        return img


# def read_frame_as_jpeg(in_file, frame_num):
#     out, err = (
#         ffmpeg.input(in_file)
#         .filter('select', 'gte(n, {})'.format(frame_num))
#         .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
#         .run(capture_stdout=True)
#     )
#     return out
#
#
# def get_video_info(in_file):
#     try:
#         probe = ffmpeg.probe(in_file)
#         video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
#         if video_stream is None:
#             print("No video stream found", file=sys.stderr)
#             sys.exit(1)
#         return video_stream
#     except ffmpeg.Error as err:
#         print(str(err.stderr, encoding='utf8'))
#         sys.exit(1)
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Enhancer')

    parser.add_argument('--border', action='store_true',
                        help='restore border images of video if true')
    parser.add_argument('--video_path', type=str, default='../dataset/DVD/test',
                        help='the path of test data')
    parser.add_argument('--model_path', type=str, default='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
                        help='the path of pretrain model')
    parser.add_argument('--result_path', type=str, default='../infer_results',
                        help='the path of result')
    parser.add_argument('--n_seq', type=int, default=3,
                        help='number of input sequences')
    parser.add_argument('--model', type=str, default='VBDE',
                        help='model type')
    parser.add_argument('--size_must_mode', type=int, default=4,
                        help='resolution mode')
    parser.add_argument('--in_rgb_range', type=int, default=255,
                        help='maximum RGB value of input video')
    parser.add_argument('--out_rgb_range', type=int, default=65535,
                        help='maximum RGB value of output video')

    args = parser.parse_args()
    video_enhancer = VideoEnhancer(args)
    video_enhancer.infer()
