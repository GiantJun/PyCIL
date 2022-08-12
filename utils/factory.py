from methods.coil import COIL
from methods.der import DER
from methods.ewc import EWC
from methods.finetune import Finetune
from methods.gem import GEM
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.finetune_replay import Replay
from methods.bic import BiC
from methods.podnet import PODNet
from methods.wa import WA


def get_trainer(config):
    name = config.method.lower()
    if name == 'icarl':
        return iCaRL(config)
    elif name == 'bic':
        return BiC(config)
    elif name == 'podnet':
        return PODNet(config)
    elif name == "lwf":
        return LwF(config)
    elif name == "ewc":
        return EWC(config)
    elif name == "wa":
        return WA(config)
    elif name == "der":
        return DER(config)
    elif name == "finetune":
        return Finetune(config)
    elif name == "finetune_replay":
        return Replay(config)
    elif name == "gem":
        return GEM(config)
    elif name == "coil":
        return COIL(config)
    else:
        assert 0
