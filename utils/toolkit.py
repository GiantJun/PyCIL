import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment):
    assert len(y_pred) == len(y_true), 'Data length error.'
    total_acc = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
    known_classes = 0
    task_acc_list = []

    # Grouped accuracy
    for cur_classes in increment:
        idxes = np.where(np.logical_and(y_true >= known_classes, y_true < known_classes + cur_classes))[0]
        task_acc_list.append(np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2))
        known_classes += cur_classes
        if known_classes >= nb_old:
            break

    return total_acc, task_acc_list


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
