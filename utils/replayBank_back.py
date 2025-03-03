import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import logging
from torch import nn
import torch
from utils.data_manager import DataManager

EPSILON = 1e-8

class ReplayBank:

    def __init__(self, config):
        self._method = config.method
        self._batch_size = config.batch_size
        self._num_workers = config.num_workers

        self._memory_size = config.memory_size
        # 有两种样本存储形式, 但都固定存储空间。一种是固定每一类数据存储样本的数量(为True时)
        # 另一种在固定存储空间中，平均分配每一类允许存储的样本数量
        self._fixed_memory = config.fixed_memory
        if self._fixed_memory:
            if config.memory_per_class == None:
                raise ValueError('if apply fix memory, memory_per_class should not be None !')
            else:
                self._memory_per_class = config.memory_per_class
        self._sampling_method = config.sampling_method # 采样的方式

        self._data_memory = [] # 第一维长度为类别数，第二维为每一类允许存放的样本数
        self._vector_memory = []
        self._valid_memory = []

        self._class_means = []
    
    def store_samplers(self, data_manager:DataManager, model, class_range):
        if self._fixed_memory:
            per_class = self._memory_per_class
        else:
            per_class = self._memory_size // (len(self._data_memory) + len(class_range))
            if len(self._data_memory) != 0:
                self.reduce_memory(per_class)

        logging.info('Constructing exemplars for the sequence of {} new classes...'.format(len(class_range)))
        for class_idx in class_range:
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            logging.info("New Class {} instance will be stored: {} => {}".format(class_idx, len(targets), per_class))
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            vectors = self._extract_vectors(model, idx_loader)
            if self._sampling_method == 'icarl':
                selected_idx = self.icarl_select(vectors, per_class)
            elif self._sampling_method == 'random':
                selected_idx = self.random_select(vectors, per_class)
            elif self._sampling_method == 'closest_to_mean':
                selected_idx = self.closest_to_mean_select(vectors, per_class)
            
            self._data_memory.append(data[selected_idx])
            self._vector_memory.append(vectors[selected_idx].cpu().numpy())
        
        logging.info('Replay Bank stored {} classes, {} samples for each class'.format(len(self._data_memory), per_class))
        self.caculate_class_mean(model, data_manager.get_dataset) #传入函数名

    def caculate_class_mean(self, model, get_dataset):
        logging.info('Calculating Class Means')
        # 重新计算旧类类中心
        class_means = []
        for class_idx, class_samples in enumerate(self._data_memory):
            idx_dataset = get_dataset([], source='train', mode='test', 
                                appendent=(class_samples, np.full(len(class_samples), class_idx)))
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            
            flip_dataset = get_dataset([], source='train', mode='flip', 
                                appendent=(class_samples, np.full(len(class_samples), class_idx)))
            flip_loader = DataLoader(flip_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

            idx_vectors = self._extract_vectors(model, idx_loader)
            flip_vectors = self._extract_vectors(model, flip_loader)

            idx_vectors = F.normalize(idx_vectors, dim=1)# 对特征向量做归一化
            flip_vectors = F.normalize(flip_vectors, dim=1)
            
            # mean = (torch.mean(idx_vectors, dim=0) + torch.mean(flip_vectors, dim=0)) /2
            mean = torch.mean(idx_vectors, dim=0)
            mean = F.normalize(mean, dim=0)
            class_means.append(mean.unsqueeze(0))
            logging.info('calculated class mean of class {}'.format(class_idx))
        
        self._class_means = torch.cat(class_means)

    def random_select(self, vectors, m):
        idxes = np.arange(vectors.shape[0])
        np.random.shuffle(idxes)
        return idxes[:m]
    
    def closest_to_mean_select(self, vectors, m):
        normalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(normalized_vector, dim=0).unsqueeze(0)
        # class_mean = F.normalize(class_mean, dim=0)
        distences = torch.cdist(normalized_vector, class_mean).squeeze()
        return torch.argsort(distences)[:m].cpu()

    def icarl_select(self, vectors, m):
        selected_idx = []
        all_idxs = list(range(vectors.shape[0]))
        nomalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(nomalized_vector, dim=0)
        for k in range(1, m+1):
            sub_vectors = nomalized_vector[all_idxs]
            S = torch.sum(sub_vectors, axis=0)
            mu_p = (sub_vectors + S) / k
            i = torch.argmin(torch.norm(class_mean-mu_p, p=2, dim=1))
            selected_idx.append(all_idxs.pop(i))
        return selected_idx

    def reduce_memory(self, m):
        for i in range(len(self._data_memory)):
            logging.info("Old class {} storage will be reduced: {} => {}".format(i, len(self._data_memory[i]), m))
            self._data_memory[i] = self._data_memory[i][:m]
            self._vector_memory[i] = self._vector_memory[i][:m]
            # logging.info("类别 {} 存储样本数为: {}".format(i, len(self._data_memory[i])))

    def KNN_classify(self, vectors=None, model=None, loader=None):
        if model != None and loader != None:
            vectors = self._extract_vectors(model, loader)
        
        vectors = F.normalize(vectors, dim=1)# 对特征向量做归一化

        dists = torch.cdist(vectors, self._class_means, p=2)
        nme_predicts = torch.argmin(dists, dim=1)
        return nme_predicts

    def get_memory(self, ret_vectors=False):
        target= []
        for class_idx, class_samples in enumerate(self._data_memory):
            target.append(np.full(len(class_samples), class_idx))

        logging.info('Replay stored samples info: stored_class={} , samples_per_class={} , total={}'.format(
                                                len(target),len(target[0]), len(target)*len(target[0])))
        
        if ret_vectors:
            return np.concatenate(self._data_memory), np.concatenate(target), np.concatenate(self._vector_memory)
        else:
            return np.concatenate(self._data_memory), np.concatenate(target)

            
    def _extract_vectors(self, model, loader):
        model.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                if isinstance(model, nn.DataParallel):
                    _vectors = model.module.extract_vector(_inputs.cuda())
                else:
                    _vectors = model.extract_vector(_inputs.cuda())

                vectors.append(_vectors)

        return torch.cat(vectors)
    