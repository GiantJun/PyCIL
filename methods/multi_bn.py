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

class multi_bn_pretrained(BaseLearner):
    def __init__(self, config):
        super().__init__(config)
        self._network_list = []
        self.bn_type = config.bn_type


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info('All params: {}'.format(count_parameters(self._network_list)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network_list, True)))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
    
        if self._cur_task == 0:
            self._network_list.append(IncrementalNet(self._convnet_type, False))
            #compare the difference between using and unusing class augmentation in first session
            self._network_list[self._cur_task].update_fc(self._cur_class)

        else:
            self._network_list.append(IncrementalNet(self._convnet_type, False))
            self._network_list[self._cur_task].update_fc(self._cur_class)
            state_dict = self._networks[self._cur_task].convnet.state_dict()

            #["default", "last", "first", "pretrained"]
            if self.bn_type == "default":
                logging.info("update_bn_with_default_setting")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
                self.reset_bn(self._networks[self._cur_task].convnet)
            elif self.bn_type == "last":
                logging.info("update_bn_with_last_model")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
            elif self.bn_type == "first":
                logging.info("update_bn_with_first_model")
                state_dict.update(self._networks[0].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
            else:
                #to be finished
                logging.info("update_bn_with_pretrained_model")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                dst_dict = torch.load("./saved_parameters/imagenet200_simsiam_pretrained_model_bn.pth")
                state_dict.update(dst_dict)
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)

        if len(self._multiple_gpus) > 1:
            self._networks[self._cur_task] = nn.DataParallel(self._networks[self._cur_task], self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._networks[self._cur_task] = self._networks[self._cur_task].module


    def _train(self, train_loader, test_loader):
        self._networks[self._cur_task].cuda()
        logging.info("parameters need grad")
        for name, param in self._networks[self._cur_task].named_parameters():
            if self._networks[self._cur_task].convnet.is_fc(name) or self._networks[self._cur_task].convnet.is_bn(name):
                logging.info(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
        if self._cur_task==0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._networks[self._cur_task].parameters()), momentum=0.9,lr=self._init_lr,weight_decay=self._init_weight_decay) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self._init_milestones, gamma=self._init_lr_decay)            
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._networks[self._cur_task].parameters()), lr=self._lrate, momentum=0.9, weight_decay=self._weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self._milestones, gamma=self._lrate_decay)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)
    

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0:
            epochs = self._init_epoch
        else:
            epochs = self._epochs
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                logits = self._network(inputs)['logits']

                fake_targets=targets-self._known_classes
                loss_clf = F.cross_entropy(logits[:,self._known_classes:], fake_targets)
                
                loss=loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)