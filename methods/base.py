import copy
import logging
import numpy as np
import torch
from torch import nn
from utils.toolkit import tensor2numpy, accuracy
from os.path import join
import logging
from backbone.inc_net import IncrementalNet
from utils.replayBank import ReplayBank
from utils.toolkit import count_parameters
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import cross_entropy

EPSILON = 1e-8

# base is finetune with or without memory_bank
class BaseLearner(object):
    def __init__(self, config, tblog):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._config = copy.deepcopy(config)
        self._network = None

        # 评价指标变化曲线
        self.cnn_task_metric_curve = None
        self.nme_task_metric_curve = None
        self.cnn_metric_curve = []
        self.nme_metric_curve = []

        self._tblog = tblog

        self._method = config.method
        self._incre_type = config.incre_type
        self._apply_nme = config.apply_nme
        self._dataset = config.dataset
        self._backbone = config.backbone
        self._seed = config.seed
        self._save_models = config.save_models

        self._memory_size = config.memory_size
        self._fixed_memory = config.fixed_memory
        self._sampling_method = config.sampling_method
        if self._fixed_memory:
            self._memory_per_class = config.memory_per_class
        self._memory_bank = None
        if (self._memory_size != None and self._fixed_memory != None and 
            self._sampling_method != None and self._incre_type == 'cil'):
            self._memory_bank = ReplayBank(self._config)
            logging.info('Memory bank created!')

        self._multiple_gpus = list(range(len(config.device.split(','))))
        self._eval_metric = config.eval_metric
        self._logdir = config.logdir
        self._opt_type = config.opt_type
        self._history_epochs = 0

        self._epochs = config.epochs
        self._init_epochs = config.epochs if config.init_epochs == None else config.init_epochs
        self._batch_size = config.batch_size
        self._num_workers = config.num_workers

    @property
    def cur_taskID(self):
        return self._cur_task

    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        if self._cur_task > 0 and self._memory_bank != None:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), 
                    source='train', mode='train', appendent=self._memory_bank.get_memory())
        else:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        
        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)


    def prepare_model(self):
        if self._network == None:
            self._network = IncrementalNet(self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        if self._incre_type == 'cil':
            self._network.update_fc(self._total_classes)
        elif self._incre_type == 'til':
            self._network.update_til_fc(self._cur_classes)
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))


    # need to be overwrite probably, base is finetune
    def incremental_train(self):
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        logging.info('-'*10 + ' Learning on task {} : {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        self._network = self._train_model(self._network, self._train_loader, self._test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train_model(self, model, train_loader, test_loader):
        self._network.cuda()
        optimizer = self._get_optimizer(self._network, self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        for epoch in range(epochs):
            model, train_acc, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler)
            test_acc = self._epoch_test(model, test_loader)
            info = ('Task {}, Epoch {}/{} => '.format(self._cur_task, epoch+1, epochs) + 
            ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses) +
            'Train_accy {:.2f}, Test_accy {:.2f}'.format(train_acc, test_acc))
            
            for i in range(int(len(train_losses)/2)):
                self._tblog.add_scalar('seed{}_train/{}'.format(self._seed, train_losses[i*2]), train_losses[i*2+1], self._history_epochs+epoch)
            self._tblog.add_scalar('seed{}_train/Acc'.format(self._seed), train_acc, self._history_epochs+epoch)
            self._tblog.add_scalar('seed{}_test/Acc'.format(self._seed), test_acc, self._history_epochs+epoch)
            logging.info(info)
        
        self._history_epochs += epochs
        return model


    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            if self._incre_type == 'cil':
                logits = model(inputs)['logits']
                loss = cross_entropy(logits, targets)
                preds = torch.max(logits, dim=1)[1]
            elif self._incre_type == 'til':
                logits = model.forward_til(inputs, self._cur_task)['logits']
                loss = cross_entropy(logits, targets - self._known_classes)
                preds = torch.max(logits, dim=1)[1] + self._known_classes
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_pred_target=False):
        cnn_correct, total = 0, 0
        cnn_pred_all, nme_pred_all, target_all = [], [], []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if self._incre_type == 'cil':
                outputs = model(inputs)
                cnn_preds = torch.max(outputs['logits'], dim=1)[1]
            elif self._incre_type == 'til':
                outputs = model.forward_til(inputs, self._cur_task)
                cnn_preds = torch.max(outputs['logits'], dim=1)[1] + self._known_classes
                
            if ret_pred_target:
                if self._memory_bank != None and self._apply_nme:
                    nme_pred = self._memory_bank.KNN_classify(outputs['features'])
                    nme_pred_all.append(tensor2numpy(nme_pred))
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            nme_pred_all = np.concatenate(nme_pred_all) if len(nme_pred) != 0 else nme_pred_all
            target_all = np.concatenate(target_all)
            return cnn_pred_all, nme_pred_all, target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc


    def eval_task(self, data_manager):
        if self._memory_bank != None:
            self._memory_bank.store_samplers(data_manager, self._network, range(self._known_classes, self._total_classes))
        
        if self.cnn_task_metric_curve == None:
            self.cnn_task_metric_curve = [[] for i in range(data_manager.nb_tasks)]
            self.nme_task_metric_curve = [[] for i in range(data_manager.nb_tasks)]

        logging.info(50*"-")
        logging.info("log {} of every task".format(self._eval_metric))
        logging.info(50*"-")
        if self._incre_type == 'cil':
            cnn_pred, nme_pred, y_true = self.get_cil_pred_target(self._network, self._test_loader)
        elif self._incre_type == 'til':
            cnn_pred, nme_pred, y_true = self.get_til_pred_target(self._network, data_manager)

        # 计算top1, 这里去掉了计算 topk 的代码
        if self._eval_metric == 'acc':
            cnn_total, cnn_task = accuracy(cnn_pred.T, y_true, self._total_classes, data_manager._increments)
        else:
            pass
        self.cnn_metric_curve.append(cnn_total)
        logging.info("CNN : {} curve of all task is {}".format(self._eval_metric, self.cnn_metric_curve))
        for i in range(len(cnn_task)):
            self.cnn_task_metric_curve[i].append(cnn_task[i])
            logging.info("CNN : task {} {} curve is {}".format(i, self._eval_metric, self.cnn_task_metric_curve[i]))
        logging.info("CNN : Average Acc: {:.2f}".format(np.mean(self.cnn_metric_curve)))
        logging.info(' ')
    
        if len(nme_pred) != 0:
            if self._eval_metric == 'acc':
                nme_total, nme_task = accuracy(nme_pred.T, y_true, self._total_classes, data_manager._increments)
            else:
                pass
            self.nme_metric_curve.append(nme_total)
            logging.info("NME : {} curve of all task is {}".format(self._eval_metric, self.nme_metric_curve))
            for i in range(len(nme_task)):
                self.nme_task_metric_curve[i].append(nme_task[i])
                logging.info("NME : task {} {} curve is {}".format(i, self._eval_metric, self.nme_task_metric_curve[i]))
            logging.info("NME : Average Acc: {:.2f}".format(np.mean(self.nme_metric_curve)))
            logging.info(' ')
        
        self._known_classes = self._total_classes
        if self._save_models:
            self.save_checkpoint('{}_{}_{}_task{}_seed{}.pkl'.format(
                self._method, self._dataset, self._backbone, self._seed), 
                self._network)

    # need to be overwrite probably
    def after_task(self):
        self._known_classes = self._total_classes
        if self._save_models:
            self.save_checkpoint('{}_{}_{}_task{}_seed{}.pkl'.format(
                self._method, self._dataset, self._backbone, self._seed), 
                self._network)
    

    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = join(self._logdir, filename)
        if state_dict != None:
            model_dict = state_dict
        else:
            model_dict = model.state_dict()
        save_dict = {'state_dict': model_dict}
        save_dict.update(self._config.get_save_config())
        torch.save(save_dict, save_path)
        logging.info('model state dict saved at: {}'.format(save_path))

    
    def get_cil_pred_target(self, model, loader):
        return self._epoch_test(model, loader, True)
    
    def get_til_pred_target(self, model, data_manager):
        known_classes = 0
        total_classes = 0
        cnn_pred_result, nme_pred_result, y_true_result = [], [], []
        for task_id in range(self._cur_task + 1):
            cur_classes = data_manager.get_task_size(task_id)
            total_classes += cur_classes
            test_dataset = data_manager.get_dataset(np.arange(known_classes, total_classes), source='test', 
                                                    mode='test')
            test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            cnn_pred, nme_pred, y_true = self._epoch_test(model, test_loader, True)
            cnn_pred_result.append(cnn_pred+known_classes)
            if len(nme_pred) > 0:
                nme_pred_result.append(nme_pred+known_classes)
            y_true_result.append(y_true)
            known_classes = total_classes

        if len(nme_pred_result) == 1:
            cnn_pred_result = cnn_pred_result[0]
            nme_pred_result = nme_pred_result[0]
            y_true_result = y_true_result[0]
        elif len(nme_pred_result) > 1:
            cnn_pred_result = np.concatenate(cnn_pred_result)
            nme_pred_result = np.concatenate(nme_pred_result)
            y_true_result = np.concatenate(y_true_result)
        else:
            cnn_pred_result = np.concatenate(cnn_pred_result)
            y_true_result = np.concatenate(y_true_result)

        return cnn_pred_result, nme_pred_result, y_true_result

    def _extract_vectors(self, model, loader):
        model.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(model, nn.DataParallel):
                _vectors = tensor2numpy(model.module.extract_vector(_inputs.cuda()))
            else:
                _vectors = tensor2numpy(model.extract_vector(_inputs.cuda()))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    

    def _get_optimizer(self, model, config, is_init):
        optimizer = None
        if is_init:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9, lr=config.init_lrate, weight_decay=config.init_weight_decay)
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lrate)
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        else:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9, lr=config.lrate, weight_decay=config.weight_decay)
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lrate)
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        return optimizer
    
    def _get_scheduler(self, optimizer, config, is_init):
        scheduler = None
        if is_init:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.init_milestones, gamma=config.init_lrate_decay)
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.init_epochs)
            else: 
                raise ValueError('No optimazer: {}'.format(config.scheduler))
        else:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.milestones, gamma=config.lrate_decay)
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs)
            else: 
                raise ValueError('No optimazer: {}'.format(config.scheduler))
        return scheduler
    