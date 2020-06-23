import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
import data


class Trainer_MOTION_NET(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_MOTION_NET, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-MOTION-NET")
        assert args.n_sequence == 2, \
            "Only support args.n_sequence=2; but get args.n_sequence={}".format(args.n_sequence)
        self.args = args

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam([{"params": self.model.parameters()}], **kwargs)

    def train(self):
        print("Now training")
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        for batch, (img1, img2, filename) in enumerate(self.loader_train):
            
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            flow = self.model(img1, img2)

            self.optimizer.zero_grad()
            loss = self.loss(img1, img2, flow)
            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch)
                ))

        self.loss.end_log(len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            total_num = 0.
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (img1, img2, filename) in enumerate(tqdm_test):

                filename = filename[1][0]
                
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                flow = self.model(img1, img2)
                warped_img2 = utils.warp(img2, flow)

                PSNR = utils.calc_psnr(img1, warped_img2, rgb_range=1.)
                total_num = total_num + 1
                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    img1, warped_img2 = utils.postprocess(img1, warped_img2,
                                                        rgb_range=self.args.rgb_range,
                                                        ycbcr_flag=False, device=self.device)
                    save_list = [img1, warped_img2]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR_iter: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

