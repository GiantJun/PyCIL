import logging
from statistics import mode
from matplotlib.pyplot import cla
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
import torch
# import medmnist
# from medmnist import INFO
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from backbone.inc_net import IncrementalNet
from utils import data_manager
from utils.toolkit import target2onehot, tensor2numpy
from backbone.linears import SimpleLinear
from utils.toolkit import count_parameters


EPSILON = 1e-8

class Multi_BN(BaseLearner):
    def __init__(self, config, tblog):
        super().__init__(config, tblog)
        self._network_list = []
        self._bn_type = config.bn_type


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
    
        if self._cur_task == 0:
            self._network_list.append(IncrementalNet(self._backbone, True, self._config.pretrain_path))
            #compare the difference between using and unusing class augmentation in first session
            self._network = self._network_list[self._cur_task]
            if self._incre_type == 'cil':
                self._network.update_fc(self._total_classes)
            elif self._incre_type == 'til':
                self._network.update_til_fc(self._total_classes)
        else:
            self._network_list.append(IncrementalNet(self._backbone, False))
            self._network = self._network_list[self._cur_task]
            if self._incre_type == 'cil':
                self._network.update_fc(self._total_classes)
            elif self._incre_type == 'til':
                self._network.update_til_fc(self._total_classes)
            state_dict = self._network.convnet.state_dict()

            #["default", "last", "first", "pretrained"]
            if self._bn_type == "default":
                logging.info("update_bn_with_default_setting")
                state_dict.update(self._network_list[self._cur_task - 1].convnet.state_dict())
                self._network.convnet.load_state_dict(state_dict)
                self.reset_bn(self._network.convnet)
            elif self._bn_type == "last":
                logging.info("update_bn_with_last_model")
                state_dict.update(self._network_list[self._cur_task - 1].convnet.state_dict())
                self._network.convnet.load_state_dict(state_dict)
            elif self._bn_type == "first":
                logging.info("update_bn_with_first_model")
                state_dict.update(self._network_list[0].convnet.state_dict())
                self._network.convnet.load_state_dict(state_dict)
            else:
                #to be finished
                logging.info("update_bn_with_pretrained_model")
                state_dict.update(self._network_list[self._cur_task - 1].convnet.state_dict())
                dst_dict = torch.load("./saved_parameters/imagenet200_simsiam_pretrained_model_bn.pth")
                state_dict.update(dst_dict)
                self._network.convnet.load_state_dict(state_dict)

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        self._network.cuda()
        for name, param in self._network.named_parameters():
            if 'fc' in name or 'bn' in name:
                logging.info('{} require grad.'.format(name))
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = self._get_optimizer(self._network, self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)