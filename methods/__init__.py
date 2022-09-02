import importlib
import logging
from os.path import join, exists
from pyexpat import model
from methods.base import BaseLearner

def get_trainer(config, tblog) -> BaseLearner:
    method_name = config.method.lower()
    if method_name in ['finetune', 'finetune_replay']:
        return BaseLearner(config, tblog)
    
    if not exists(join('methods',method_name+'.py')):
        raise ValueError('Method Python File {}.py do not exist!'.format(method_name))
    
    model = None
    model_filename = 'methods.' + method_name
    modellib = importlib.import_module(model_filename)
    for cls_name, cls in modellib.__dict__.items():
        if cls_name.lower() == method_name:
            model = cls

    if model is None:
        raise ValueError('Method class {} do not exist!'.format(method_name))
    
    logging.info('Trainer {} created!'.format(method_name))
    return model(config, tblog)

# def get_trainer(config, tblog) -> BaseLearner:
#     name = config.method.lower()
#     if name in ['finetune', 'finetune_replay']:
#         return BaseLearner(config, tblog)
#     elif name == 'joint':
#         return Joint(config, tblog)
#     elif name == 'icarl':
#         return iCaRL(config, tblog)
#     elif name == 'bic':
#         return BiC(config, tblog)
#     elif name == 'podnet':
#         return PODNet(config, tblog)
#     elif name == "lwf":
#         return LwF(config, tblog)
#     elif name == "ewc":
#         return EWC(config, tblog)
#     elif name == "wa":
#         return WA(config, tblog)
#     elif name == "der":
#         return DER(config, tblog)
#     elif name == "gem":
#         return GEM(config, tblog)
#     elif name == "coil":
#         return COIL(config, tblog)
#     elif name == "multi_bn":
#         return Multi_BN(config, tblog)
#     elif name == "multi_bn_selectt":
#         return Multi_BN_selectT(config, tblog)
#     elif name == "cnnprompt_joint":
#         return CNNPrompt_Joint(config, tblog)
#     else:
#         raise ValueError('Unknown method {}'.format(name))