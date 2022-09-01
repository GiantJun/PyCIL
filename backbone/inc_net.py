import copy
from typing import Iterable
import torch
from torch import nn, rand
import logging
from backbone.cifar_resnet import resnet32
import torchvision.models as torch_models
from backbone.ucir_cifar_resnet import resnet32 as cosine_resnet32
from backbone.ucir_resnet import resnet18 as cosine_resnet18
from backbone.ucir_resnet import resnet34 as cosine_resnet34
from backbone.ucir_resnet import resnet50 as cosine_resnet50
from backbone.linears import SplitCosineLinear, CosineLinear
from backbone.cifar_resnet_cbam import resnet18_cbam as resnet18_cbam
from typing import Callable


def get_backbone(backbone_type, pretrained=False, pretrain_path=None, normed=False) -> nn.Module:
    name = backbone_type.lower()
    net = None
    if name in torch_models.__dict__.keys():
        net = torch_models.__dict__[name](pretrained=pretrained)
    elif name == 'resnet32':
        net = resnet32()
    elif name == 'cosine_resnet18':
        net = cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        net = cosine_resnet32()
    elif name == 'cosine_resnet34':
        net = cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        net = cosine_resnet50(pretrained=pretrained)
    elif name == 'resnet18_cbam':
        net = resnet18_cbam(normed=normed)
    else:
        raise NotImplementedError('Unknown type {}'.format(backbone_type))
    logging.info('Created {} !'.format(name))

    # 载入自定义预训练模型
    if pretrain_path != None and pretrained:
        pretrained_dict = torch.load(pretrain_path)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = torch.load(pretrain_path)['state_dict']
        state_dict = net.state_dict()
        logging.info('special keys in load model state dict: {}'.format(pretrained_dict.keys()-state_dict.keys()))
        for key in (pretrained_dict.keys() & state_dict.keys()):
            state_dict[key] = pretrained_dict[key]
        net.load_state_dict(state_dict)

        logging.info("loaded pretrained_dict_name: {}".format(pretrain_path))

    return net


class IncrementalNet(nn.Module):

    def __init__(self, backbone_type, pretrained, pretrain_path=None):
        super(IncrementalNet, self).__init__()
        self.feature_extractor = get_backbone(backbone_type, pretrained, pretrain_path)
        if 'resnet' in backbone_type:
            self._feature_dim = self.feature_extractor.fc.in_features
        else:
            raise ValueError('{} did not support yet!'.format(backbone_type))
        self.feature_extractor.fc = nn.Sequential()
        self.fc = None
        self.fc_til = None
        self.output_features = {'features': torch.empty(0)}

    @property
    def feature_dim(self):
        return self._feature_dim

    def extract_vector(self, x):
        features = self.feature_extractor(x)
        return features

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.fc(features)
        self.output_features['features'] = features
        return out, self.output_features
    
    def forward_til(self, x, task_id):
        features = self.feature_extractor(x)
        out = self.fc_til[task_id](features)
        self.output_features['features'] = features
        return out, self.output_features

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            logging.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            logging.info('Created classifier head with output dim {}'.format(nb_classes))
        del self.fc
        self.fc = fc

    def update_til_fc(self, nb_classes):
        if self.fc_til == None:
            self.fc_til = nn.ModuleList([])
        self.fc_til.append(self.generate_fc(self.feature_dim, nb_classes))

    def generate_fc(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
    

class IncrementalNet_fm(IncrementalNet):
    '''IncrementalNet returns feature map from specific layers'''

    def __init__(self, backbone_type, pretrained, pretrain_path=None, layer_names: Iterable[str]=[]):
        '''
        layers_name can be ['layer1','layer2','layer3','layer4']
        '''
        super(IncrementalNet_fm, self).__init__(backbone_type, pretrained, pretrain_path)
        self.layer_names = layer_names
        self.output_features = {layer: torch.empty(0) for layer in layer_names}

        model_dict = dict([*self.feature_extractor.named_modules()]) 
        for layer_id in layer_names:
            layer = model_dict[layer_id]
            layer.register_forward_hook(self.save_output_features(layer_id))
    
    def save_output_features(self, layer_id: str) -> Callable:
        def hook(module, input, output):
            self.output_features[layer_id] = output
        return hook


# class IncrementalNet(BaseNet):

#     def __init__(self, convnet_type, pretrained, pretrain_path=None, gradcam=False):
#         super(IncrementalNet, self).__init__(convnet_type, pretrained, pretrain_path)
#         self.gradcam = gradcam
#         if hasattr(self, 'gradcam') and self.gradcam:
#             self._gradcam_hooks = [None, None]
#             self.set_gradcam_hook()

#     def weight_align(self, increment):
#         weights=self.fc.weight.data
#         newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
#         oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
#         meannew=torch.mean(newnorm)
#         meanold=torch.mean(oldnorm)
#         gamma=meanold/meannew
#         print('alignweights,gamma=',gamma)
#         self.fc.weight.data[-increment:,:]*=gamma

#     def forward(self, x):
#         x = self.convnet(x)
#         out = self.fc(x['features'])
#         out.update(x)
#         if hasattr(self, 'gradcam') and self.gradcam:
#             out['gradcam_gradients'] = self._gradcam_gradients
#             out['gradcam_activations'] = self._gradcam_activations

#         return out
    
#     def forward_til(self, x, task_id):
#         x = self.convnet(x)
#         out = self.fc_til[task_id](x['features'])
#         out.update(x)
#         if hasattr(self, 'gradcam') and self.gradcam:
#             out['gradcam_gradients'] = self._gradcam_gradients
#             out['gradcam_activations'] = self._gradcam_activations

#         return out

#     def unset_gradcam_hook(self):
#         self._gradcam_hooks[0].remove()
#         self._gradcam_hooks[1].remove()
#         self._gradcam_hooks[0] = None
#         self._gradcam_hooks[1] = None
#         self._gradcam_gradients, self._gradcam_activations = [None], [None]

#     def set_gradcam_hook(self):
#         self._gradcam_gradients, self._gradcam_activations = [None], [None]

#         def backward_hook(module, grad_input, grad_output):
#             self._gradcam_gradients[0] = grad_output[0]
#             return None

#         def forward_hook(module, input, output):
#             self._gradcam_activations[0] = output
#             return None

#         self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
#         self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)


class CosineIncrementalNet(IncrementalNet):

    def __init__(self, convnet_type, pretrained, nb_proxy=1):
        super().__init__(convnet_type, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)
        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(IncrementalNet):
    def __init__(self, convnet_type, pretrained, bias_correction=False):
        super().__init__(convnet_type, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        if self.bias_correction:
            logits = out['logits']
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
            out['logits'] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(DERNet,self).__init__()
        self.convnet_type=convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features
    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out=self.fc(features) #{logics: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"]

        out.update({"aux_logits":aux_logits,"features":features})
        return out        
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def update_fc(self, nb_classes):
        if len(self.convnets)==0:
            self.convnets.append(get_backbone(self.convnet_type))
        else:
            self.convnets.append(get_backbone(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()
    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

class SimpleCosineIncrementalNet(IncrementalNet):
    
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data=self.fc.sigma.data
            if nextperiod_initialization is not None:
                
                weight=torch.cat([weight,nextperiod_initialization])
            fc.weight=nn.Parameter(weight)
        del self.fc
        self.fc = fc
        

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

class CNNPromptNet(IncrementalNet):
    def __init__(self, convnet_type, pretrained, pretrain_path=None, prompt_size=32, gamma=1.):
        super().__init__(convnet_type, pretrained, pretrain_path)
        self.gamma = gamma
        self.prompt = nn.Parameter(rand((1,3,prompt_size,prompt_size)))

    def forward(self, x):
        prompt = self.prompt.expand(x.shape[0], 3, -1, -1)
        x_out = self.convnet(x)['features']
        prompt_out = self.convnet(prompt)['features']
        out = self.fc(x_out+self.gamma*prompt_out)
        out.update({'features': x_out+self.gamma*prompt_out})
        
        return out
    
    def freeze(self):
        for name, param in self.named_parameters():
            if 'fc' in name or 'prompt' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.eval()
        return self