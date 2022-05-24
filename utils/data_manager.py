import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iTinyImageNet200, Skin7, SD_198, Skin8, MyMedMnist


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        """
        dataset_name: 数据集名称
        shuffle: 是否对类别标签进行重排列, 即随机类别到达顺序
        init_cls: 初始阶段训练类别数, 对于含子数据集的数据集无效
        increment: 每阶段增加类别的数量, 对于含子数据集的数据集无效
        """
        self.dataset_name = dataset_name
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        if hasattr(idata, '_dataset_inc'):  # for datasets which have sub-datasets, likes MedMNistS
            self._increments = idata._dataset_inc
            logging.info('SubDataset order: {}'.format(idata._dataset_info))
            self._class_order = idata.class_order
        else:
            # Order
            order = idata.class_order
            if shuffle:
                np.random.seed(seed)
                order = np.random.permutation(len(order)).tolist()
            else:
                order = idata.class_order
            self._class_order = order
            logging.info('class order: {}'.format(self._class_order))
            assert init_cls <= len(self._class_order), 'No enough classes.'
            self._increments = [init_cls]
            while sum(self._increments) + increment < len(self._class_order):
                self._increments.append(increment)
            offset = len(self._class_order) - sum(self._increments)
            if offset > 0:
                self._increments.append(offset)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    @property
    def nb_tasks(self):
        """
        作用: 获得数据到达的总批次数
        """
        return len(self._increments)
    
    @property
    def total_classes(self):
        """
        作用: 获得本数据集包含的所有类别数
        """
        return sum(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        """
        作用: 获取指定类别范围的数据
        indices: 想要获取类别数据的类标号范围
        source: 可选值为 train 或 test, 确定是训练集还是测试集
        mode: 可选值为 train 或 flip(水平翻转) 或 test, 数据增广的方式
        appendent: 额外数据及其标签列表, 在获得指定 indices 范围类别数据外, 额外加入的数据
        ret_data: 布尔值, 是否范围数据及标签列表
        """
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)


    def _select(self, x, y, low_range, high_range):
        """
        作用: 返回 x, y 中指定范围 (low_range, high_range) 中的数据
        """
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

# map class y to its index of order
# y = [0, 1, 2, 3, 4]
# order = [1, 3, 0, 2, 4]
# result = [2, 0, 3, 1, 4] : 0 -> 2, 1 -> 0, 2 -> 3, 3 -> 1, 4 -> 4 
def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10()
    elif name == 'cifar100':
        return iCIFAR100()
    elif name == 'imagenet1000':
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "tinyimagenet":
        return iTinyImageNet200()
    elif name == "skin7":
        return Skin7()
    elif name == "skin8":
        return Skin8()
    elif name == "sd198" or name == 'skin40':
        return SD_198()
    elif name == "mymedmnist":
        return MyMedMnist()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
