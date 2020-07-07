import torch

import data
import model
import loss
import option
from trainer.trainer_cdvd_tsp import Trainer_CDVD_TSP
from trainer.trainer_vbde import Trainer_VBDE
from trainer.trainer_motion_net import  Trainer_MOTION_NET
from logger import logger


if __name__ == '__main__':
    args = option.args
    torch.manual_seed(args.seed)
    chkp = logger.Logger(args)

    if args.task == 'VideoBDE':
        print("Selected task: {}".format(args.task))
        model = model.Model(args, chkp)
        loss = loss.Loss(args, chkp) if not args.test_only else None
        loader = data.Data(args)
        t = Trainer_VBDE(args, loader, model, loss, chkp)
        val_pnsr = 0
        while not t.terminate():
            if args.scheduler == 'plateau':
                t.scheduler.step(val_pnsr)
            else:
                t.scheduler.step()
            t.train()
            val_pnsr = t.test()
            t.loader_train = loader.update_train_loader(args)

    elif args.task == 'OpticalFlow':
        print("Selected task: {}".format(args.task))
        model = model.Model(args, chkp)
        loss = loss.Loss(args, chkp) if not args.test_only else None
        loader = data.Data(args)
        t = Trainer_MOTION_NET(args, loader, model, loss, chkp)
        while not t.terminate():
            t.train()
            t.test()
            t.loader_train = loader.update_train_loader(args)
    else:
        raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

    chkp.done()
