import logging
import numpy as np
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from backbone.inc_net import CNNPromptNet
from utils.toolkit import count_parameters


EPSILON = 1e-8


class CNNPrompt_Joint(BaseLearner):
    def __init__(self, config, tblog):
        super().__init__(config, tblog)
        if self._incre_type != 'cil':
            raise ValueError('Joint learning is a class incremental method obviously!')

    def prepare_task_data(self, data_manager):
        self._cur_task = data_manager.nb_tasks-1
        self._cur_class = data_manager.total_classes
        self._total_classes = self._known_classes + self._cur_class

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)


    def prepare_model(self):
        if self._network == None:
            self._network = CNNPromptNet(self._config.backbone, self._config.pretrained, self._config.pretrain_path, self._config.prompt_size, self._config.gamma)
        self._network.update_fc(self._total_classes)
        if self._config.freeze:
            self._network.freeze()
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))