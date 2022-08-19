import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from backbone.inc_net import DERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from torch.nn.functional import cross_entropy

EPSILON = 1e-8

class DER(BaseLearner):
    def __init__(self, config, tblog):
        super().__init__(config, tblog)
        if self._incre_type != 'cil':
            raise ValueError('DER is a class incremental method!')

    def prepare_model(self):
        if self._network == None:
            self._network = DERNet(self._config.backbone, self._config.pretrained)
        self._network.update_fc(self._total_classes)

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def incremental_train(self):
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network = self._train_model(self._network, self._train_loader, self._test_loader)
        if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
        else:
            self._network.weight_align(self._total_classes-self._known_classes)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        losses_clf = 0.
        losses_aux = 0.
        
        model.train()
        if len(self._multiple_gpus) > 1:
            network = self._network.module
        else:
            network = self._network
        network.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                network.convnets[i].eval()

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)['logits']
            logits, aux_logits = outputs["logits"], outputs["aux_logits"]
            
            loss_clf = cross_entropy(logits, targets)
            aux_targets = targets.clone()
            aux_targets = torch.where(aux_targets-self._known_classes+1>0, aux_targets-self._known_classes+1, 0)
            loss_aux = F.cross_entropy(aux_logits,aux_targets)
            loss = loss_clf + loss_aux

            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            losses_aux += loss_aux.item()
            losses_clf += loss_clf.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', losses_clf/len(train_loader), losses_aux/len(train_loader)]
        return model, train_acc, train_loss