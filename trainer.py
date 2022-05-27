import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import datetime

def train(args):
    seed_list = copy.deepcopy(args['seed'])

    os.environ['CUDA_VISIBLE_DEVICES']=args['device']

    for seed in seed_list:
        args['seed'] = seed
        _train(args)


def _train(args):
    try:
        os.makedirs("logs/{}".format(args['method']))
    except:
        pass
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logfilename = 'logs/{}/{}_{}_{}_{}_{}_{}_{}_{}'.format(args['method'], args['prefix'], args['seed'], args['method'], args['backbone'],
                                                args['dataset'], args['init_cls'], args['increment'], nowTime)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    _set_random()
    # _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['method'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        model.eval_task(data_manager)
        model.after_task()


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    # log hyperparameter
    logging.info(30*"=")
    logging.info("log_hyperparameters")
    logging.info(30*"-")
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
    logging.info(30*"=")
