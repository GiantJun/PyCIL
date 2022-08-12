import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os
from PIL import Image
import sys

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        # or replay os.environ['xxx'] with './data/'
        train_dataset = datasets.cifar.CIFAR10(os.environ['DATA'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(os.environ['DATA'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iTinyImageNet200(iData):
    use_path = False

    train_trsf = [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.230, 0.227, 0.226]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        self.root_dir = os.path.join(os.environ["PUBLICDATASETS"], "tiny-imagenet-200")
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        self._create_class_idx_dict_train()
        self._create_class_idx_dict_val()

        self._make_dataset()

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))

        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        train_images = []
        test_images = []
        train_list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        test_list_of_dirs = ["images"]

        for tgt in train_list_of_dirs:
            dirs = os.path.join(self.train_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        #add picture loading code
                        item = (pil_loader(path), self.class_to_tgt_idx[tgt])
                        train_images.append(item)
        
        for tgt in test_list_of_dirs:
            dirs = os.path.join(self.val_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        item = (pil_loader(path), self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        test_images.append(item)

        self.train_data, self.train_targets = split_images_labels(train_images)
        self.test_data, self.test_targets = split_images_labels(test_images)

class SD_198(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224,scale=(0.3,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        ]
    
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.592, 0.479, 0.451], std=[0.265, 0.245, 0.247]),
    ]

    class_order = np.arange(40).tolist()

    def getdata(self, fn):
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(os.environ["SD198DATASETS"], 'images', temp[0]))
                targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        train_dir = os.path.join(os.environ["SD198DATASETS"], "main_classes_split/train_1.txt")
        test_dir = os.path.join(os.environ["SD198DATASETS"], "main_classes_split/val_1.txt")

        self.train_data, self.train_targets = self.getdata(train_dir)
        self.test_data, self.test_targets = self.getdata(test_dir)

class Skin7(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224,scale=(0.3,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        ]
    
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7720412, 0.54642653, 0.56280327], std=[0.13829944, 0.1553769, 0.17688483]),
    ]

    class_order = np.arange(7).tolist()

    def getdata(self, fn):
        print(fn)
        csvfile = pd.read_csv(fn)
        raw_data = csvfile.values

        data = []
        targets = []
        for path, label in raw_data:
            data.append(os.path.join(os.environ["SKIN7DATASETS"],
                                     "ISIC2018_Task3_Training_Input", path))
            targets.append(label)

        return np.array(data), np.array(targets)

    def download_data(self):
        train_dir = os.path.join(os.environ["SKIN7DATASETS"], "split_data/split_data_1_fold_train.csv")
        test_dir = os.path.join(os.environ["SKIN7DATASETS"], "split_data/split_data_1_fold_test.csv")

        self.train_data, self.train_targets = self.getdata(train_dir)
        self.test_data, self.test_targets = self.getdata(test_dir)

class Skin8(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224,scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        ]
    
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7720412, 0.54642653, 0.56280327], std=[0.13829944, 0.1553769, 0.17688483]),
    ]

    class_order = np.arange(8).tolist()

    def getdata(self, fn):
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                data.append(os.path.join(os.environ["SKIN8DATASETS"], temp[0]))
                targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        train_dir = os.path.join(os.environ["SKIN8DATASETS"], "train_skin8_500.txt")
        test_dir = os.path.join(os.environ["SKIN8DATASETS"], "test_skin8_500.txt")

        self.train_data, self.train_targets = self.getdata(train_dir)
        self.test_data, self.test_targets = self.getdata(test_dir)

class MyMedMnist(iData):

    # 由于 MedMnist 中有些子数据集是多标签或者是3D的，这里没有使用
    # 以下表示中，字符为子数据集名字, 数字为子数据集中包含的类别数
    _dataset_info = [('organamnist',11), ('dermamnist',7), ('pneumoniamnist',2), ('bloodmnist',8), ('pathmnist',9),
                        ('breastmnist',2), ('tissuemnist',8), ('octmnist',4)]
    use_path = False

    train_trsf = [
        transforms.Resize((32, 32)),
        transforms.RandomCrop((32,32),padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        ]
    
    test_trsf = [
        transforms.Resize((32, 32)),
    ]

    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(51).tolist()

    _dataset_inc = [data_flag[1] for data_flag in _dataset_info]

    def getdata(self, src_dir):
        train_data = np.array([])
        train_targets = np.array([])
        test_data = np.array([])
        test_targets = np.array([])
        
        known_class = 0
        for data_flag in self._dataset_info:
            npz_file = np.load(os.path.join(src_dir, "{}.npz".format(data_flag[0])))

            train_imgs = npz_file['train_images']
            train_labels = npz_file['train_labels']

            test_imgs = npz_file['test_images']
            test_labels = npz_file['test_labels']

            if len(train_imgs.shape) == 3:
                train_imgs = np.expand_dims(train_imgs, axis=3)
                train_imgs = np.repeat(train_imgs, 3, axis=3)

                test_imgs = np.expand_dims(test_imgs, axis=3)
                test_imgs = np.repeat(test_imgs, 3, axis=3)

            train_labels = train_labels + known_class
            train_labels = train_labels.astype(np.uint8)

            test_labels = test_labels + known_class
            test_labels = test_labels.astype(np.uint8)
            
            train_data = np.concatenate([train_data, train_imgs]) if len(train_data) != 0 else train_imgs
            train_targets = np.concatenate([train_targets, train_labels]) if len(train_targets) != 0 else train_labels
            

            test_data = np.concatenate([test_data, test_imgs]) if len(test_data) != 0 else test_imgs
            test_targets = np.concatenate([test_targets, test_labels]) if len(test_targets) != 0 else test_labels

            known_class = len(np.unique(train_targets))
        
        return train_data, np.squeeze(train_targets,1), test_data, np.squeeze(test_targets,1)


    def download_data(self):
        # 在我们划分的数据集中，后面这一项有几种选项可选
        # mymedmnist_1_in_5: 对不均衡的 PathMNIST,OCTMNIST, TissueMNIST, OrganAMNIST, 将其随机下采样为其原来的1/5, 其余不变
        # mymedmnist_1000_300: 对所有子数据集, 均随机采样成样本数分别为 1000,300 的训练集和测试集
        # origin: MedMNIST 原始数据集
        src_dir = os.path.join(os.environ["MYMEDMNIST"], "mymedmnist_1_in_5")
        self.train_data, self.train_targets, self.test_data, self.test_targets = self.getdata(src_dir)
        # print(self.train_data.shape)
        # print(self.test_data.shape)

        print(len(np.unique(self.train_targets)))
        print(len(np.unique(self.test_targets)))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return np.array(img)