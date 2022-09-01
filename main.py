from utils.config import Config
from utils.toolkit import set_logger
import copy
import os
import torch
import logging
from utils.data_manager import DataManager
# from utils import factory
import methods


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    config = Config()
    seed_list = copy.deepcopy(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES']=config.device

    tblog = set_logger(config, True)
    try:
        for seed in seed_list:
            temp_config = copy.deepcopy(config)
            temp_config.seed = seed
            set_random(seed)
            data_manager = DataManager(temp_config.dataset, temp_config.shuffle, temp_config.seed, temp_config.init_cls, temp_config.increment)
            temp_config.update({'total_class_num':data_manager.total_classes, 'task_num':data_manager.nb_tasks})
            temp_config.print_config()
            trainer = methods.get_trainer(temp_config, tblog)

            # for task in range(data_manager.nb_tasks):
            while trainer.cur_taskID < data_manager.nb_tasks - 1:
                trainer.prepare_task_data(data_manager)
                trainer.prepare_model()
                trainer.incremental_train()
                trainer.eval_task(data_manager)
                trainer.after_task()

    except Exception as e:
        logging.error(e, exc_info=True, stack_info=True)
