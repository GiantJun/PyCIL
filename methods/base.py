import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
import torch.nn.functional as F
from scipy.spatial.distance import cdist

EPSILON = 1e-8
# batch_size = 64


class BaseLearner(object):
    def __init__(self, config):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        # 评价指标变化曲线
        self.cnn_task_metric_curve = None
        self.nme_task_metric_curve = None
        self.cnn_metric_curve = []
        self.nme_metric_curve = []

        self._memory_size = config.memory_size
        self._fixed_memory = config.fixed_memory
        if self._fixed_memory:
            self._memory_per_class = config.memory_per_class
        self._multiple_gpus = list(range(len(config.device.split(','))))
        self._eval_metric = config.eval_metric

        self._epochs = config.epochs
        self._lrate = config.lrate
        self._milestones = config.milestones
        self._lrate_decay = config.lrate_decay
        self._batch_size = config.batch_size
        self._weight_decay = config.weight_decay
        self._num_workers = config.num_workers

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), 'Exemplar size error.'
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, 'Total classes is 0'
            return (self._memory_size // self._total_classes)

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': self._network.state_dict(),
        }
        torch.save(save_dict, '{}_{}.pkl'.format(filename, self._cur_task))

    def after_task(self):
        pass


    def eval_task(self, data_manager):
        if self.cnn_task_metric_curve == None:
            self.cnn_task_metric_curve = [[] for i in range(data_manager.nb_tasks)]
            self.nme_task_metric_curve = [[] for i in range(data_manager.nb_tasks)]

        cnn_pred, nme_pred, y_true = self._eval_cnn_nme(self.test_loader)
        
        logging.info(50*"-")
        logging.info("log {} of every task".format(self._eval_metric))
        logging.info(50*"-")

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
    
        if len(nme_pred) > 0:
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


    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)
    
    def _eval_cnn_nme(self, loader):
        self._network.eval()
        cnn_pred, nme_pred, y_true = [], [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = self._network(inputs)
            cnn_predicts = torch.argmax(outputs['logits'], dim=1)
            cnn_pred.append(tensor2numpy(cnn_predicts))
            if hasattr(self, '_class_means'):
                vectors = tensor2numpy(outputs['features'])
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                scores = cdist(vectors, self._class_means)
                nme_predicts = np.argmin(scores, axis=1)
                nme_pred.append(nme_predicts)
            y_true.append(targets)

        return np.concatenate(cnn_pred), np.concatenate(nme_pred) if len(nme_pred)!=0 else nme_pred, \
                np.concatenate(y_true)  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.cuda()))
            else:
                _vectors = tensor2numpy(self._network.extract_vector(_inputs.cuda()))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test',
                                                   appendent=(selected_exemplars, exemplar_targets))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
