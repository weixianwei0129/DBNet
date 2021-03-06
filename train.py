import os
import shutil
import time

import yaml
import argparse
import numpy as np
from easydict import EasyDict

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models.rescspnet import ResNetCSP
from losses.db_loss import DBLoss
from datasets.polygon import PolygonDataSet

from post_processing.SynthResult import SynthResult, draw_boxes_on_img

from common import one_cycle, color_str, norm_img

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_img(images):
    concat = []
    for img in images:
        if len(img.size()) == 2 or img.size()[0] == 1:
            img = torch.stack([img] * 3, dim=0)
        concat.append(norm_img(img))
    return torch.cat(concat, dim=2)


class Trainer(object):
    def __init__(self, opt):
        self.text_out = SynthResult()
        self.opt = opt
        # Specifying the disk address
        workspace = os.path.join(opt.project, opt.name)
        self.ckpt_folder = os.path.join(workspace, 'ckpt')
        if not os.path.exists(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)
            print(f"create a path {color_str(self.ckpt_folder)}.")
        # copy config file
        shutil.copy(opt.cfg, os.path.join(workspace, 'config.yaml'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = EasyDict(yaml.safe_load(open(opt.cfg)))
        self.model = ResNetCSP(self.cfg.model)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.db_loss = DBLoss()

        self.loss_flag = np.inf

        # Load data
        train_dataset = PolygonDataSet(self.cfg.data, 'train')
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=4,  # 可用GPU数量的四倍
            drop_last=True,
            pin_memory=True
        )

        test_dataset = PolygonDataSet(self.cfg.data, 'test')
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=3
        )

        # 创建优化器
        if self.cfg.train.optimizer == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.lr)
        elif self.cfg.train.optimizer == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=self.cfg.train.lr, momentum=0.99)
        else:
            raise ValueError(f'error optimizer: {self.cfg.train.optimizer}')
        if opt.linear_lr:
            lr_factor = lambda x: (1 - x / (opt.epochs - 1)) ** 0.9
        else:
            lr_factor = one_cycle(1, self.cfg.train.lf, opt.epochs)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_factor)
        # 混合精度 FP32 + FP16
        self.scaler = amp.GradScaler(enabled=device)

        self.writer = SummaryWriter(log_dir=workspace, flush_secs=10)

        self.epoch = 0
        self.minimum_loss = np.inf

    def _train_one_epoch(self):
        self.model.train()

        batch_iter = 0
        total_num = len(self.train_loader)
        total_batch_iters = round(total_num * self.cfg.data.batch_size / self.cfg.train.batch_size)

        # 每次更新opt需要更新这个值
        cur_batch_size = 0
        loss_collection = np.zeros((4,))

        self.optimizer.zero_grad()
        for index, data in enumerate(self.train_loader):
            cur_batch_size += self.cfg.data.batch_size
            img = data['img'].to(device)
            shrunk_segment = data['shrunk_segment'].to(device)
            threshold_map = data['threshold'].to(device)
            train_mask = data['train_mask'].to(device)

            # forward
            predict = self.model(img)

            # backward
            loss_info = self.db_loss(
                predict,
                shrunk_segment,
                threshold_map,
                train_mask
            )

            loss_collection += np.array([
                loss_info['synth'].item(),
                loss_info['score'].item(),
                loss_info['binary'].item(),
                loss_info['threshold'].item(),
            ])

            self.scaler.scale(loss_info['synth']).backward()

            if cur_batch_size >= self.cfg.train.batch_size or index == total_num - 1:
                # 更新优化器
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Log 整个一次epoch,显示5次
                if batch_iter % round(total_batch_iters / 5) == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    loss_value = loss_collection / cur_batch_size
                    loss_collection = np.zeros((4,))
                    step = self.epoch * total_num + batch_iter

                    # ====Summery====
                    idx = np.random.randint(0, img.shape[0])
                    images = [
                        img[idx],
                        shrunk_segment[idx],
                        threshold_map[idx],
                        shrunk_segment[idx] - threshold_map[idx],
                        predict[idx][0],
                        predict[idx][1]
                    ]
                    self.writer.add_image('train/img', concat_img(images), step)
                    self.writer.add_scalar('train/loss/synth', loss_value[0], step)
                    self.writer.add_scalar('train/loss/score', loss_value[1], step)
                    self.writer.add_scalar('train/loss/binary', loss_value[2], step)
                    self.writer.add_scalar('train/loss/threshold', loss_value[3], step)
                    self.writer.add_scalar('train/learning_rate', lr, step)
                    output_log = f"[{time.asctime(time.localtime())}] " \
                                 f"[{batch_iter + 1:4d}/{total_batch_iters:4d}] " \
                                 f"Loss(synth/score/binary/threshold): " \
                                 f"{loss_value[0]:.4f}/{loss_value[1]:.4f}/" \
                                 f"{loss_value[2]:.4f}/{loss_value[3]:.4f}"
                    print(output_log)

                # 训练进行了一个batch
                batch_iter += 1
                cur_batch_size = 0
                loss_collection = np.zeros((4,))

    @torch.no_grad()
    def _do_a_test(self):
        self.model.eval()

        total_num = len(self.test_loader)
        loss_collection = np.zeros((4,))
        images = []
        for index, data in enumerate(self.test_loader):
            img = data['img'].to(device)
            shrunk_segment = data['shrunk_segment'].to(device)
            threshold_map = data['threshold'].to(device)
            train_mask = data['train_mask'].to(device)
            # forward
            predict = self.model(img)
            if not index % round(total_num / 5):
                images += [
                    img[0],
                    shrunk_segment[0],
                    threshold_map[0],
                    train_mask[0],
                    predict[0][0],
                    predict[0][1],

                ]
                self.writer.add_image(
                    'test/img',
                    concat_img(images),
                    self.epoch
                )

            loss_info = self.db_loss(
                predict, shrunk_segment, threshold_map, train_mask
            )
            loss_collection += np.array([
                loss_info['synth'].item(),
                loss_info['score'].item(),
                loss_info['binary'].item(),
                loss_info['threshold'].item(),
            ])

        loss_value = loss_collection / total_num
        self.writer.add_scalar('test/loss/synth', loss_value[0], self.epoch)
        self.writer.add_scalar('test/loss/score', loss_value[1], self.epoch)
        self.writer.add_scalar('test/loss/binary', loss_value[2], self.epoch)
        self.writer.add_scalar('test/loss/threshold', loss_value[3], self.epoch)
        output_log = f"[TEST] [{time.asctime(time.localtime())}] " \
                     f"Loss(synth/score/binary/threshold): " \
                     f"{loss_value[0]:.4f}/{loss_value[1]:.4f}/" \
                     f"{loss_value[2]:.4f}/{loss_value[3]:.4f}"
        print(color_str(output_log, 'red'))
        return loss_value[0].item()

    def _save_model(self, loss):
        state = dict(
            epoch=self.epoch,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            loss=loss
        )
        save_path = os.path.join(self.ckpt_folder, "last.pt")
        torch.save(state, save_path)
        print("save a model at ", color_str(save_path, 'yellow'))
        if loss < self.loss_flag:
            self.loss_flag = loss
            save_path = os.path.join(self.ckpt_folder, "best.pt")
            torch.save(state, save_path)
            print(f"{color_str('update', 'yellow')} a best "
                  f"model at ", color_str(save_path, 'yellow'))

    def run(self):
        # select train type :
        # 1. load pretrain model; 2. resume train; 3. train from scratch
        if self.opt.pretrain:
            pretrain_file = self.opt.weights
            assert os.path.isfile(pretrain_file), 'Error: no pretrained weights found!'
            checkpoint = torch.load(pretrain_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Fine tuning from {color_str('pretrained model : ', 'red')} "
                  f"{color_str(pretrain_file)}")
        elif self.opt.resume:
            checkpoint_path = os.path.join(self.ckpt_folder, "last.pt")
            assert os.path.isfile(checkpoint_path), \
                f'Error: no checkpoint file found at {color_str(checkpoint_path)}'

            checkpoint = torch.load(checkpoint_path)
            self.epoch = checkpoint.get('epoch', 0)
            self.loss_flag = checkpoint.get('loss', np.inf)
            self.minimum_loss = checkpoint.get('best_loss', np.inf)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"{color_str('restore', 'red')} from {color_str(checkpoint_path)}")
        else:
            if not os.path.exists(self.ckpt_folder):
                os.makedirs(self.ckpt_folder)
            print(f"Train model from {color_str('scratch', 'red')} "
                  f"and save at {color_str(self.ckpt_folder)}")

        while self.epoch < self.opt.epochs:
            print(f"[{self.epoch} | {self.opt.epochs}]")
            self._train_one_epoch()
            self.epoch += 1
            if self.epoch % 50 == 0:
                loss = self._do_a_test()
                if self.epoch > 0.2 * self.opt.epochs:
                    self._save_model(loss)


def main(opt):
    trainer = Trainer(opt)
    trainer.run()


def parse_opt():
    parser = argparse.ArgumentParser(description="DBNet")
    parser.add_argument('--cfg', type=str, default='config/xxx.yaml', help="Description from README.md.")
    parser.add_argument('--epochs', type=int, default=300, help='Total epoch during training.')
    parser.add_argument('--project', type=str, default='', help='Project path on disk')
    parser.add_argument('--name', type=str, default='vx.x.x', help='Name of train model')
    parser.add_argument('--linear_lr', action='store_false', help='Whether use linear lr')
    parser.add_argument('--pretrain', action='store_true', help='Whether to use a pre-training model')
    parser.add_argument('--weights', type=str, default='xx.pt', help="Pretrain the model's path on disk")
    parser.add_argument('--resume', action='store_true',
                        help="Whether to resume, and find the `last.pt` file as weights")
    hyp = parser.parse_args()
    return hyp


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
