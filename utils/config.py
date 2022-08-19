import argparse
from numpy import int64
import yaml
import logging

def load_yaml(settings_path):
    args = {}
    with open(settings_path) as data_file:
        param = yaml.load(data_file, Loader=yaml.FullLoader)
    args.update(param['basic'])
    if not 'test' in args['method']: # 测试不需要指定训练参数
        dataset = args['dataset']
        backbone = args['backbone']
        if 'options' in param:
            args.update(param['options'][dataset][backbone])
    if 'special' in param:
        args.update(param['special'])
    return args

class Config:

    def __init__(self):
        parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
        parser.add_argument('--config', type=str, default=None, help='yaml file of settings.')

        # basic config
        self.basic_config_names = ['device', 'seed', 'num_workers', 'dataset', 'split_for_valid', 'backbone', 'method', 'pretrained',
            'incre_type', 'pretrained', 'pretrain_path', 'freeze', 'save_models', 'eval_metric', 'init_cls', 'increment']
        self.special_config_names = ['T', 'memory_size', 'fixed_memory', 'sampling_method', 'split_ratio', 'lambda',
            'fishermax', 'lambda_c_base', 'lambda_f_base', 'nb_proxy', 'ft_epochs', 'ft_lrate']

        self.init_overwrite_names = []
        self.load_overwrite_names = []

        parser.add_argument('--device', nargs='+', type=int, default=None, help='GPU ids, e.g. 0 (for single gpu) or 0 1 2 (for multi gpus)')
        parser.add_argument('--seed', nargs='+', type=int, default=None, help='random seed for the programe, 0 (for single seed) or 0 1 2 (run in seed 0 1 2 respectively)')
        parser.add_argument('--num_workers', type=int, default=None, help='CPUs for dataloader')
        
        parser.add_argument('--dataset', type=str, default=None, help='dataset to be used')
        parser.add_argument('--shuffle', type=str, default=None, help='shuffle class order')
        parser.add_argument('--split_for_valid', type=bool, default=None, help='whether to split training set to true training set and valid set') # 赋初值为 None 相当于 False
        
        parser.add_argument('--backbone', type=str, default=None, help='backbone to train')
        
        parser.add_argument('--method', type=str, default=None, help='methods to apply')

        parser.add_argument('--apply_nme', type=bool, default=None, help='whether apply nme to classify') 

        parser.add_argument('--incre_type', type=str, default=None, help='Incremental type e.t. cil or til')
        parser.add_argument('--pretrained', type=bool, default=None, help='whether use pretrained network weights to initial the network')
        parser.add_argument('--pretrain_path', type=str, default=None, help='prtrained network weights path to load')
        parser.add_argument('--freeze', type=bool, default=None, help='freeze the feature extractor')
        parser.add_argument('--save_models', type=bool, default=None, help='save trained models weights')
        parser.add_argument('--eval_metric', type=str, default=None, help='evaluate metrics option')

        parser.add_argument('--init_cls', type=str, default=None, help='init class num for the training')
        parser.add_argument('--increment', type=str, default=None, help='increment class num for the training')

        # training config
        parser.add_argument('--init_epochs', type=int, default=None, help='init training epochs')
        parser.add_argument('--init_lrate', type=float, default=None, help='init training learning rate')
        parser.add_argument('--init_scheduler', type=str, default=None, help='init learning rate decay method')
        parser.add_argument('--init_milestones', type=int, default=None, help='init milestones for training')
        parser.add_argument('--init_lrate_decay', type=float, default=None, help='init training learning rate decay')
        parser.add_argument('--init_weight_decay', type=int, default=None, help='init weight decay for training')

        parser.add_argument('--epochs', type=int, default=None, help='training epochs')
        parser.add_argument('--batch_size', type=int, default=None, help='batch size for training')
        parser.add_argument('--lrate', type=float, default=None, help='training learning rate')
        parser.add_argument('--opt_type', type=str, default=None, help='optimizer for training')
        parser.add_argument('--weight_decay', type=float, default=None, help='weight decay for sgd')
        parser.add_argument('--scheduler', type=str, default=None, help='learning rate decay method')
        parser.add_argument('--milestones', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
        parser.add_argument('--lrate_decay', type=float, default=None, help='for multi step learning rate decay scheduler')
        parser.add_argument('--criterion', type=str, default=None, help='loss function, e.g. ce, focal')

        # special config
        parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')

        parser.add_argument('--memory_size', type=int, default=None, help='memory size for memory buffer')
        parser.add_argument('--fixed_memory', type=bool, default=None, help='fix memory size for each class')
        parser.add_argument('--memory_per_class', type=bool, default=None, help='memory size per class for fixed memory')
        parser.add_argument('--sampling_method', type=str, default=None, help='sampler methods option for memory buffer')

        parser.add_argument('--split_ratio', type=float, default=None, help='split ratio for bic') # bic

        parser.add_argument('--lambda', type=float, default=None, help='lambda for ewc or lwf') # ewc or lwf
        parser.add_argument('--fishermax', type=float, default=None, help='fishermax for ewc') # ewc

        parser.add_argument('--lambda_c_base', type=float, default=None, help='lambda_c_base for podnet') # podnet
        parser.add_argument('--lambda_f_base', type=float, default=None, help='lambda_f_base for podnet') # podnet
        parser.add_argument('--nb_proxy', type=int, default=None, help='nb_proxy for podnet') # podnet
        parser.add_argument('--ft_epochs', type=int, default=None, help='ft_epochs for podnet') # podnet
        parser.add_argument('--ft_lrate', type=float, default=None, help='ft_lrate for podnet') # podnet

        parser.add_argument('--bn_type', type=str, default=None, help='mode for multi_bn method, e.g. default, last, first')

        # multi-bn
        parser.add_argument('--multi_bn_type', type=str, default=None, help='different type of multi-bn, e.g. default,last,first,pretrained')

        for name, value in vars(parser.parse_args()).items():
            setattr(self, name, value)
        
        if self.config != None:
            init_config = load_yaml(self.config)
            for name, value in init_config.items():
                if getattr(self, name) == None:
                    setattr(self, name, value)
                    self.init_overwrite_names.append(name)
            print('Loaded config file: {}'.format(self.config))
    
    def get_save_config(self) -> dict:
        result = {}
        for item in self.basic_config_names:
            result.update({item:getattr(self, item)})
        for item in self.special_config_names:
            result.update({item:getattr(self, item)})
        return result
    
    def load_saved_config(self, init_dict: dict) -> None:
        if 'state_dict' in init_dict:
            init_dict.pop('state_dict')
        for key, value in init_dict.items():
            if (not hasattr(self,key)) or getattr(self, key) == None:
                setattr(self, key, value)
                self.load_overwrite_names.append(key)
                
        if 'test' in self.method: # 包含 test emsemble_test_avg, emsemble_test_vote
            self.kfold = 1
            self.pretrained = True
            self.split_for_valid = False
        elif self.method == 'retrain':
            self.save_name = 'retrain_' + self.save_name

    def update(self, update_dict: dict) -> None:
        for key, value in update_dict.items():
            setattr(self, key, value)
    
    # @property
    # def is_two_stage_method(self) -> bool:
    #     return self.method in ['simclr', 'mocoV2', 'sup_simclr', 'bal_sup_moco']
    
    def print_config(self) -> None:
        logging.info(30*"=")
        logging.info("log hyperparameters in seed {}".format(self.seed))
        logging.info(30*"-")
        for name, value in vars(self).items():
            if not name in ['basic_config_names', 'special_config_names', 'init_overwrite_names', 'load_overwrite_names'] and not getattr(self, name) == None:
                logging.info('{}: {}'.format(name, value))
        logging.info(30*"=")
        logging.info('Inital configs overwrited by yaml config file: {}'.format(self.init_overwrite_names))
        logging.info('Inital configs overwrited by loaded pkl configs: {}'.format(self.load_overwrite_names))
        