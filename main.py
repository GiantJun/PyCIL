from utils.config import Config
from utils.toolkit import set_logger
import copy
import os
import torch
import logging
from utils.data_manager import DataManager
from utils import factory
from utils.toolkit import count_parameters

def set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    config = Config()
    seed_list = copy.deepcopy(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES']=config.device

    for seed in seed_list:
        temp_config = copy.deepcopy(config)
        tblog = set_logger(temp_config, True)
        temp_config.seed = seed
        set_random()
        config.print_config()
        data_manager = DataManager(temp_config.dataset, temp_config.shuffle, temp_config.seed, temp_config.init_cls, temp_config.increment)
        trainer = factory.get_trainer(temp_config)

        for task in range(data_manager.nb_tasks):
            logging.info('All params: {}'.format(count_parameters(trainer._network)))
            logging.info('Trainable params: {}'.format(count_parameters(trainer._network, True)))
            trainer.incremental_train(data_manager)
            trainer.eval_task(data_manager)
            trainer.after_task()
