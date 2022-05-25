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


def get_model(model_name, args):
    name = model_name.lower()
    if name == 'icarl':
        return iCaRL(args)
    elif name == 'bic':
        return BiC(args)
    elif name == 'podnet':
        return PODNet(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "wa":
        return WA(args)
    elif name == "der":
        return DER(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "finetune_replay":
        return Replay(args)
    elif name == "gem":
        return GEM(args)
    elif name == "coil":
        return COIL(args)
    else:
        assert 0
