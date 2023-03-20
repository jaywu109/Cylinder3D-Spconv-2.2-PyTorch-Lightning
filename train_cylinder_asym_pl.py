# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import os
import argparse
import random
import numpy as np
import math
import os
import time
import argparse
import sys
import torch
# torch.set_float32_matmul_precision('high')


from utils.metric_util import per_class_iu, fast_hist_crop_torch
from dataloader.pc_dataset import get_SemKITTI_label_name
from dataloader.dataset_semantickitti import  collate_fn_BEV
from builder import data_builder_pl, model_builder, loss_builder
from config.config_pl import load_config_data
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
import pytorch_lightning as pl
from multiprocessing import get_context

import warnings

warnings.filterwarnings("ignore")

class Cylinder3D(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

        seed=123 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.save_hyperparameters(configs)        

        self.dataset_config = configs['dataset_params']
        self.train_dataloader_config = configs['train_data_loader']
        self.val_dataloader_config = configs['val_data_loader']
        self.val_batch_size = self.val_dataloader_config['batch_size']
        self.train_batch_size = self.train_dataloader_config['batch_size']
        self.model_config = configs['model_params']
        self.train_hypers = configs['train_params']
        self.gpu_num = len(self.train_hypers['gpus'])
        self.grid_size = self.model_config['output_shape']   

        SemKITTI_label_name = get_SemKITTI_label_name(self.dataset_config["label_mapping"])
        self.unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        self.unique_label_str = [SemKITTI_label_name[x] for x in self.unique_label + 1]

        self.model = model_builder.build(self.model_config)     

        self.loss_func, self.lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                    num_class=self.model_config['num_class'], ignore_label=self.dataset_config['ignore_label'])

        if self.gpu_num > 1:
            self.effective_lr = math.sqrt((self.gpu_num)*(self.train_batch_size)) * self.train_hypers["base_lr"]
        else:
            self.effective_lr = math.sqrt(self.train_batch_size) * self.train_hypers["base_lr"]

        print(self.effective_lr)
        self.save_hyperparameters({'effective_lr': self.effective_lr})

        self.train_dataset, self.val_dataset = data_builder_pl.build(self.dataset_config,
                                                                        self.train_dataloader_config,
                                                                        self.val_dataloader_config,
                                                                        grid_size=self.grid_size) 
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def setup(self, stage=None):
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        seed = self.global_rank
       
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('local seed:', seed)
        ##############################################      

    def train_dataloader(self):          
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.train_dataloader_config["batch_size"],
                                           collate_fn=collate_fn_BEV,
                                           shuffle=self.train_dataloader_config["shuffle"],
                                           num_workers=self.train_dataloader_config["num_workers"],
                                           pin_memory=True,
                                           drop_last=True,
                                           multiprocessing_context=get_context('fork')
                                           )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.val_dataloader_config["batch_size"],
                                           collate_fn=collate_fn_BEV,
                                           shuffle=self.val_dataloader_config["shuffle"],
                                           num_workers=self.val_dataloader_config["num_workers"],
                                           pin_memory=True,
                                           multiprocessing_context=get_context('fork')
                                           )
    
    def forward(self, pt_fea_ten, vox_ten, batch_size):
        outputs = self.model(pt_fea_ten, vox_ten, batch_size)
        return outputs


    def training_step(self, batch, batch_idx):
        _, vox_label, grid, _, pt_fea = batch
        pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in
                            pt_fea]
        vox_ten = [torch.from_numpy(i).to(self.device) for i in grid]
        label_tensor = vox_label.type(torch.LongTensor).to(self.device)   
        batch_size = len(pt_fea)

        predict_vox_labels = self(pt_fea_ten, vox_ten, batch_size)
        loss = self.lovasz_softmax(torch.nn.functional.softmax(predict_vox_labels), label_tensor, ignore=0) + self.loss_func(
            predict_vox_labels, label_tensor)              
        
        train_loss_gather = self.all_gather(loss).mean()

        if self.global_rank == 0:
            self.log('train_loss_step', train_loss_gather, prog_bar=True, rank_zero_only=True)
            self.logger.experiment.add_scalar('train_loss_step', train_loss_gather, self.global_step)   

            with torch.no_grad():
                self.training_step_outputs.append({'loss': train_loss_gather.cpu().detach()})

        return loss

    def on_training_epoch_end(self):
        if self.global_rank == 0:
            train_loss_mean = torch.stack([output['loss'] for output in self.training_step_outputs]).mean()   
            self.logger.experiment.add_scalar('train_loss_epoch', train_loss_mean, self.current_epoch)   
            self.log('train_loss_epoch', train_loss_mean, prog_bar=True, logger=False)
            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        _, vox_label, grid, pt_labs, pt_fea = batch
        pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in
                            pt_fea]
        vox_ten = [torch.from_numpy(i).to(self.device) for i in grid]
        label_tensor = vox_label.type(torch.LongTensor).to(self.device)   
        batch_size = len(pt_fea)

        predict_vox_labels = self(pt_fea_ten, vox_ten, batch_size)
        loss = self.lovasz_softmax(torch.nn.functional.softmax(predict_vox_labels), label_tensor, ignore=0) + self.loss_func(
            predict_vox_labels, label_tensor)    
        
        predict_vox_labels = torch.argmax(predict_vox_labels, dim=1)
        hist_list = []
        for count, _ in enumerate(grid):
            hist_list.append(fast_hist_crop_torch(predict_vox_labels[
                                                count, grid[count][:, 0], grid[count][:, 1],
                                                grid[count][:, 2]], torch.tensor(pt_labs[count]).type(torch.LongTensor).to(self.device),
                                            self.unique_label))
            
        hist_sum = sum(hist_list)    
        if self.gpu_num > 1:   
            hist_sum = sum(self.all_gather(hist_sum))
        validation_loss_gather = self.all_gather(loss).mean()

        if self.global_rank == 0:   
                self.validation_step_outputs.append({'loss':validation_loss_gather.cpu().detach(), 'hist': hist_sum.cpu().detach()})

    def on_validation_epoch_end(self):
        if self.global_rank == 0:

            loss_list = []
            hist_list = []
            for output in self.validation_step_outputs:
                loss_list.append(output['loss'])
                hist_list.append(output['hist'])

            val_loss_mean = torch.stack(loss_list).mean() 
            hist_sum = sum(hist_list)

            self.log('val_loss', val_loss_mean, prog_bar=True, logger=False)
            self.logger.experiment.add_scalar('val_loss', val_loss_mean, self.global_step)   

            iou = per_class_iu(hist_sum.numpy())

            val_miou = torch.tensor(np.nanmean(iou) * 100).to(self.device)  
            self.logger.experiment.add_scalar('val_miou', val_miou, self.global_step)   

            self.validation_step_outputs.clear()

        else:
            val_miou = torch.tensor(0).to(self.device)  
        
        if self.gpu_num > 1:
            val_miou = sum(self.all_gather(val_miou))

        self.log('val_miou', val_miou, prog_bar=True, logger=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.effective_lr, eps=1e-4, weight_decay=self.train_hypers['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return {'optimizer':optimizer, 'lr_scheduler':scheduler}


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_pl.yaml')
    args = parser.parse_args()

    config_path = args.config_path
    configs = load_config_data(config_path)    
    train_params = configs['train_params']
    logdir = train_params['logdir']

    model = Cylinder3D(configs)

    gpus = train_params['gpus']
    gpu_num = len(gpus)
    print(gpus)

    train_data_num = len(model.train_dataset)
    effective_train_batch_size = configs['train_data_loader']['batch_size'] * gpu_num
    train_steps_per_epoch = int(train_data_num // effective_train_batch_size)
    val_interval_fraction = train_params['checkpoint_every_n_epochs']

    if train_params['mixed_fp16']:
        precision = '16'
    else:
        precision = '32'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    checkpoint = ModelCheckpoint(dirpath=f'{logdir}',
                                 save_last=True,
                                 save_top_k=5,
                                 mode='max',
                                 every_n_train_steps=int(train_steps_per_epoch*val_interval_fraction),
                                #  every_n_epochs=1,
                                 filename="{epoch:02d}-{step}-{val_miou:.3f}",
                                 monitor="val_miou")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir,
                                             name=None,
                                             version='')

    if gpu_num > 1:
        strategy='ddp'
        use_distributed_sampler = True
    else:
        strategy = 'auto'
        use_distributed_sampler = False

    # resume = '/public/home/wudj/Cylinder3D_uda/logs_2/epoch=09-step=9243-val_loss_epoch=1.7565454245.ckpt'
    trainer = pl.Trainer(
        # limit_val_batches=10,
        # resume_from_checkpoint=resume,
        val_check_interval=val_interval_fraction,
        max_epochs=train_params['max_num_epochs'],
        devices=gpus,
        num_nodes=1,
        precision = precision,
        accelerator='gpu',
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        logger=tb_logger,
    )


    trainer.fit(model)    
