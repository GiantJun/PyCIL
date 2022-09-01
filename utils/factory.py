from methods.base import BaseLearner
from methods.joint import Joint
from methods.coil import COIL
from methods.der import DER
from methods.ewc import EWC
from methods.base import BaseLearner
from methods.gem import GEM
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.bic import BiC
from methods.podnet import PODNet
from methods.wa import WA
from methods.multi_bn import Multi_BN
from methods.multi_bn_selectT import Multi_BN_selectT
from methods.cnnPrompt_joint import CNNPrompt_Joint


def get_trainer(config, tblog) -> BaseLearner:
    name = config.method.lower()
    if name in ['finetune', 'finetune_replay']:
        return BaseLearner(config, tblog)
    elif name == 'joint':
        return Joint(config, tblog)
    elif name == 'icarl':
        return iCaRL(config, tblog)
    elif name == 'bic':
        return BiC(config, tblog)
    elif name == 'podnet':
        return PODNet(config, tblog)
    elif name == "lwf":
        return LwF(config, tblog)
    elif name == "ewc":
        return EWC(config, tblog)
    elif name == "wa":
        return WA(config, tblog)
    elif name == "der":
        return DER(config, tblog)
    elif name == "gem":
        return GEM(config, tblog)
    elif name == "coil":
        return COIL(config, tblog)
    elif name == "multi_bn":
        return Multi_BN(config, tblog)
    elif name == "multi_bn_selectt":
        return Multi_BN_selectT(config, tblog)
    elif name == "cnnprompt_joint":
        return CNNPrompt_Joint(config, tblog)
    else:
        raise ValueError('Unknown method {}'.format(name))
