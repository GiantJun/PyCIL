import importlib
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
    
    return model(config, tblog)