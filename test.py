from typing import Callable
from torchvision import models
import torch



input = torch.rand((1,3,224,224))
a = models.resnet18()
features = {layer: torch.empty(0) for layer in ['layer1','layer2','layer3','layer4']}

def save_features(layer_id: str) -> Callable:
    def func(module, input, output):
        features[layer_id] = output
    return func

for layer_id in ['layer1','layer2','layer3','layer4']:
    layer = dict([*a.named_modules()])[layer_id]
    layer.register_forward_hook(save_features(layer_id))

print(a(input).shape)

print({name: output.shape for name, output in features.items()})

