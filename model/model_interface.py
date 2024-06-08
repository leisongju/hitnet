# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import sys
import pytorch_lightning as pl
from model.loss_file import GemoLoss, Whole_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize, TwoSlopeNorm

def norm_pic(array):
    array_min = array.min()
    array_max = array.max()
    normalized_array = (array - array_min) / (array_max - array_min)
    return normalized_array

def calculate_norm(vmin, vmax):
    include_zero = False
    """根据vmin和vmax的值计算并返回相应的norm对象"""
    if vmin >= 0:
        norm = Normalize(vmin=vmin, vmax=vmax)
        include_zero = 1
    elif vmax <= 0:
        norm = Normalize(vmin=vmin, vmax=vmax)
        include_zero = -1
    else:
        vcenter = 0
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        include_zero = 0
    return norm, include_zero

def show_disp(raw, predictions, save_path, epoch):
    pic_nums = 32  # 固定为32行
    subplot_cols = 3  # 仅绘制左图、右图和视差图
    plt.figure(figsize=(subplot_cols * 5, pic_nums * 5))
    
    titles = ['Left Image', 'Right Image', 'Disparity Map']
    
    for i in range(pic_nums):
        idx = i
        if idx < raw.shape[0]:
            left = norm_pic(raw[idx, 0:1, :, :]).detach().cpu().numpy()
            right = norm_pic(raw[idx, 1:2, :, :]).detach().cpu().numpy()
            disp = predictions[idx, 0, :, :].detach().cpu().numpy()
            
            left = np.clip(np.power(left, 1/2), 0, 1)
            right = np.clip(np.power(right, 1/2), 0, 1)

            disp_min = disp.min()
            disp_max = disp.max()
            vmin = disp_min
            vmax = disp_max
            
            norm, disp_error = calculate_norm(vmin, vmax)
            
            for col in range(subplot_cols):
                plt.subplot(pic_nums, subplot_cols, subplot_cols*i + col + 1)
                if col == 0:  # Left Image
                    plt.imshow(np.squeeze(left), cmap='gray')
                elif col == 1:  # Right Image
                    plt.imshow(np.squeeze(right), cmap='gray')
                elif col == 2:  # Disparity Map
                    if disp_error < 0:
                        im_disp = plt.imshow(disp, cmap='Blues', norm=norm)
                    elif disp_error > 0:
                        im_disp = plt.imshow(disp, cmap='Reds', norm=norm)
                    else:
                        im_disp = plt.imshow(disp, cmap='bwr', norm=norm)
                    plt.colorbar(im_disp, ax=plt.gca(), fraction=0.046, pad=0.04)
                    plt.title(f'Disparity Map\nMean: {disp.mean():.2f}')
                    
                plt.title(titles[col])
                plt.axis('off')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, f'disparity_epoch_{epoch}.png'), bbox_inches='tight')
    plt.close()



class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.validation_step_outputs = []
        self.last_batch = None

    def forward(self, left, right):
        return self.model(left, right)

    def training_step(self, batch, batch_idx):
        left, right = batch
        out = self(left, right)
        loss = self.loss_function(out, left, right, train=True)
        # 逐个记录字典中的每个值
        for key, value in loss.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        left, right = batch
        out = self(left, right)
        loss = self.loss_function(out, left, right, train=False)

        # 逐个记录字典中的每个值
        for key, value in loss.items():
            self.log(f'{key}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append(loss) 
        self.last_batch = (left, right, out)


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    # def on_validation_epoch_end(self):
    #     # Make the Progress Bar leave there
    #     self.print('')
    
    def on_train_epoch_end(self):
        print('')
        sys.stdout.flush()

    def on_validation_epoch_end(self):
        # 获取并打印每个epoch的损失值
        avg_losses = {f'val_{key}': torch.stack([x[key] for x in self.validation_step_outputs]).mean() for key in self.validation_step_outputs[0].keys()}
        print(f"Epoch {self.current_epoch}: {avg_losses}")

        if self.last_batch:
            left, right, predictions = self.last_batch
            save_path = self.logger.log_dir  # 获取日志目录
            if (self.current_epoch+1) % 10 == 0:
                show_disp(torch.cat([left, right],dim=1), predictions, save_path, self.current_epoch)



        # 清空保存的输出，以便下一个epoch使用
        self.validation_step_outputs.clear()
        sys.stdout.flush()

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'gemo':
            self.loss_function = GemoLoss()
        elif loss == 'warp':
            self.loss_function = Whole_loss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)


    