import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from methods.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torch.nn.functional import cross_entropy

EPSILON = 1e-8

class iCaRL(BaseLearner):

    def __init__(self, config, tblog):
        super().__init__(config, tblog)
        self._T = config.T
        self._old_network = None
        if self._incre_type != 'cil':
            raise ValueError('iCaRL is a class incremental method!')

    def _train_model(self, model, train_loader, test_loader):
        if self._old_network is not None:
            self._old_network.cuda()
        return super()._train_model(model, train_loader, test_loader)

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)['logits']
            
            loss_clf = cross_entropy(logits, targets)
            if self._cur_task == 0:
                loss = loss_clf
            else:
                loss_kd = self._KD_loss(logits[:,:self._known_classes],self._old_network(inputs)["logits"],self._T)
                loss = loss_clf + loss_kd

            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = losses/len(train_loader)
        return model, train_acc, train_loss

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()
        
    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred/T, dim=1)
        soft = torch.softmax(soft/T, dim=1)
        return -1 * torch.mul(soft, pred).sum()/pred.shape[0]