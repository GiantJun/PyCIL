import logging
from statistics import mode
from matplotlib.pyplot import cla
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm
import torch
# import medmnist
# from medmnist import INFO
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from backbone.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from backbone.linears import SimpleLinear
from sklearn.metrics import confusion_matrix


EPSILON = 1e-8

class multi_bn_pretrained(BaseLearner):
    def __init__(self, config):
        super().__init__(config)
        self._networks = []

        self._init_epoch = config.init_epochs
        self._init_lr = config.init_lrate
        self._init_milestones = config.init_milestones
        self._init_lr_decay = config.init_lrate_decay
        self._init_weight_decay = config.init_weight_decay

    def after_task(self):
        self._known_classes = self._total_classes

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        torch.save(self._networks[self._cur_task].state_dict(),
                        os.path.join(self._save_dir, "task_{}_seed_{}.pth".format(self._cur_task, self._seed)))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._acc_of_every_task.append([])
        self._mcr_of_every_task.append([])

        if data_manager is not None:
            self._cur_class = data_manager.get_task_size(self._cur_task)
        elif self._dataset == "MedMinist":
            data_flag = self._dataset_order[self._cur_task]
            info = INFO[data_flag]
            self._cur_class  = len(info['label'])
        
        self._total_classes = self._known_classes + self._cur_class
        self._networks.append(IncrementalNet(self._convnet_type, False))

        dst_key = self._get_key(self._convnet_type)

        if self._cur_task == 0:
            #load pretrained model
            state_dict = self._networks[self._cur_task].convnet.state_dict()
            logging.info("{}running_mean before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
            logging.info("{}weight before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
            logging.info("{}bias before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))

            if self._dataset == "sd198":
                # pretrained_dict_name = "./saved_parameters/sd198_resnet18_pretrained_1.pth"
                pretrained_dict_name = "./saved_parameters/sd198_simsiam_model_18_224.pth"
                pretrained_dict = torch.load(pretrained_dict_name)
            elif self._dataset == "cifar100":
                if self._convnet_type == "resnet32":
                    pretrained_dict_name = "./saved_parameters/imagenet200_simsiam_model_32.pth"
                    pretrained_dict = torch.load(pretrained_dict_name)
                elif self._convnet_type == "resnet18_cbam":
                    # pretrained_dict_name = "./saved_parameters/imagenet200_resnet18_cbam_pretrained.pth"
                    pretrained_dict_name = "./saved_parameters/imagenet200_simsiam_pretrained_model.pth"
                    pretrained_dict = torch.load(pretrained_dict_name)
            elif self._dataset == "MedMinist":
                pretrained_dict_name = "./saved_parameters/imagenet200_simsiam_pretrained_model.pth"
                # pretrained_dict_name = "./saved_parameters/imagenet200_resnet18_cbam_pretrained.pth"
                pretrained_dict = torch.load(pretrained_dict_name)
            
            state_dict.update(pretrained_dict)
            self._networks[self._cur_task].convnet.load_state_dict(state_dict)

            logging.info("pretrained_dict_name: {}".format(pretrained_dict_name))

            logging.info("{}running_mean after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
            logging.info("{}weight after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
            logging.info("{}bias after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))

            #compare the difference between using and unusing class augmentation in first session
            self._networks[self._cur_task].update_fc(self._cur_class)

        else:
            self._networks[self._cur_task].update_fc(self._cur_class)
            state_dict = self._networks[self._cur_task].convnet.state_dict()
            logging.info("{}running_mean before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
            logging.info("{}weight before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
            logging.info("{}bias before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))

            #["default", "last", "first", "pretrained"]
            if bn_type == "default":
                logging.info("update_bn_with_default_setting")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
                self.reset_bn(self._networks[self._cur_task].convnet)
            elif bn_type == "last":
                logging.info("update_bn_with_last_model")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
            elif bn_type == "first":
                logging.info("update_bn_with_first_model")
                state_dict.update(self._networks[0].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
            else:
                #to be finished
                logging.info("update_bn_with_pretrained_model")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                dst_dict = torch.load("./saved_parameters/imagenet200_simsiam_pretrained_model_bn.pth")
                state_dict.update(dst_dict)
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
    
            logging.info("{}running_mean after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
            logging.info("{}weight after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
            logging.info("{}bias after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))

        if self._dataset != "MedMinist":
            logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        else:
            logging.info('Learning on {}, the num of classes is {}'.format(data_flag, self._cur_class))


        if self._dataset != "MedMinist":
            # Loader
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                    mode='train')
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='test', 
                                                    mode='test')
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            DataClass = getattr(mymedmnist, info['python_class'])
            # load the data
            # set as_rgb true can change image to rgb

            train_trsf = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop((32,32),padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.24705882352941178),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])

            test_trsf = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])

            train_dataset = DataClass(split='train', transform=train_trsf, download=False, as_rgb=True)
            self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            test_dataset = DataClass(split='test', transform=test_trsf, download=False, as_rgb=True)
            # encapsulate data into dataloader form
            self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._networks[self._cur_task] = nn.DataParallel(self._networks[self._cur_task], self._multiple_gpus)
        
        self._train(self._networks[self._cur_task], self.train_loader, self.test_loader)

        logging.info("{}running_mean after training: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
        logging.info("{}weight after training: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
        logging.info("{}bias after training: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))

        if self._dataset != "MedMinist":
            self.calculate_acc_mcr(data_manager)
        else:
            self.calculate_acc_mcr()

    def _train(self, model, train_loader, test_loader):
        model.to(self._device)
        
        if self._cur_task == 0:
            if fix_parameter:
                logging.info("parameters need grad")
                for name, param in model.named_parameters():
                    if model.convnet.is_fc(name) or model.convnet.is_bn(name):
                        logging.info(name)
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                if optim_type == "adam":
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate_init, weight_decay=weight_decay_init)
                else:
                    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate_init, momentum=0.9, weight_decay=weight_decay_init)  # 1e-3
            
            else:
                if optim_type == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=lrate_init, weight_decay=weight_decay_init)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay_init)  # 1e-3
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init, gamma=lrate_decay_init)
        else:
            logging.info("parameters need grad")
            for name, param in model.named_parameters():
                if model.convnet.is_fc(name) or model.convnet.is_bn(name):
                    logging.info(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            if optim_type == "adam":
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(model, train_loader, test_loader, optimizer, scheduler)

    def reset_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.reset_parameters()

    def _update_representation(self, model, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0:
            epochs_num = epochs_init
        else:
            epochs_num = epochs

        prog_bar = tqdm(range(epochs_num))
        #if temp < 1, it will make the output of softmax sharper
        # temp = 0.1
        for _, epoch in enumerate(prog_bar):
            model.train()
            losses = 0.
            correct, total = 0, 0
            if self._dataset != "MedMinist":
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    if self._cur_task == 0:
                        if class_aug:
                            inputs, targets = self.classAug(inputs, targets)
                        logits = model(inputs)['logits']
                    else:
                        logits = model(inputs)['logits']

                    if self._dataset != "MedMinist":
                        loss = nn.CrossEntropyLoss()(logits/temp, targets - self._known_classes)
                    else:
                        loss = nn.CrossEntropyLoss()(logits/temp, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    # acc
                    _, preds = torch.max(logits, dim=1)
                    if self._dataset != "MedMinist":
                        correct += preds.eq((targets - self._known_classes).expand_as(preds)).cpu().sum()
                    else:
                        correct += preds.eq((targets).expand_as(preds)).cpu().sum()
                    total += len(targets)
            else:
                for i, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.squeeze().to(self._device)

                    if self._cur_task == 0:
                        if class_aug:
                            inputs, targets = self.classAug(inputs, targets)
                        logits = model(inputs)['logits']
                    else:
                        logits = model(inputs)['logits']

                    if self._dataset != "MedMinist":
                        loss = nn.CrossEntropyLoss()(logits/temp, targets - self._known_classes)
                    else:
                        loss = nn.CrossEntropyLoss()(logits/temp, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    # acc
                    _, preds = torch.max(logits, dim=1)
                    if self._dataset != "MedMinist":
                        correct += preds.eq((targets - self._known_classes).expand_as(preds)).cpu().sum()
                    else:
                        correct += preds.eq((targets).expand_as(preds)).cpu().sum()
                    total += len(targets)
            
            if self._cur_task == 0 and epoch == epochs_num - 1 and class_aug:
                weight = model.fc.weight.data
                bias = model.fc.bias.data
                in_feature = model.fc.in_features
                model.fc = SimpleLinear(in_feature, self._total_classes)
                model.fc.weight.data = weight[:self._total_classes]
                model.fc.bias.data = bias[:self._total_classes]
                print("The num of total classes is {}".format(self._total_classes))

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(model, test_loader)
            # if epoch == epochs_num - 1:
            #     self._task_acc.append(round(test_acc, 2))
            
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs_num, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

            logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        if self._dataset != "MedMinist":
            for i, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    outputs = model(inputs)['logits']
                predicts = torch.max(outputs, dim=1)[1]

                if self._dataset != "MedMinist":
                    correct += (predicts.cpu() == (targets - self._known_classes)).sum()
                else:
                    correct += (predicts.cpu() == targets).sum()
                total += len(targets)
        else:
            for i, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                targets = targets.squeeze()
                with torch.no_grad():
                    outputs = model(inputs)['logits']
                predicts = torch.max(outputs, dim=1)[1]

                if self._dataset != "MedMinist":
                    correct += (predicts.cpu() == (targets - self._known_classes)).sum()
                else:
                    correct += (predicts.cpu() == targets).sum()
                total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    #at most, the num of samples will be 5 times of origin
    def classAug(self, x, y, alpha=20.0, mix_times=4):  # mixup based
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(mix_times):
            #Returns a random permutation of integers 
            index = torch.randperm(batch_size).to(self._device)
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.generate_label(y[i].item(), y[index][i].item())
                    lam = np.random.beta(alpha, alpha)
                    if lam < 0.4 or lam > 0.6:
                        lam = 0.5
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.to(self._device).long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        return x, y
    
    def generate_label(self, y_a, y_b):
        if self._old_network == None:
            y_a, y_b = y_a, y_b
            #make sure y_a < y_b
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            #calculate the sum of arithmetic sequence and then sum the bias
            label_index = ((2 * self._total_classes - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
        else:
            y_a = y_a - self._known_classes
            y_b = y_b - self._known_classes
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = int(((2 * self._cur_class - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
        return label_index + self._total_classes
    
    def calculate_mean_class_recall(self, y_pred, y_true):
        """ Calculate the mean class recall for the dataset X """
        cm = confusion_matrix(y_true, y_pred)
        right_of_class = np.diag(cm)
        num_of_class = cm.sum(axis=1)
        mcr = np.around((right_of_class*100 / num_of_class).mean(), decimals=2)
        return mcr

    #take careful! MCR may be wrongly calculated because the label of Medmnist!
    def calculate_acc_mcr(self, data_manager=None):
        logging.info("Calculating ACC and MCR")
        known_classes = 0
        total_classes = 0
        correct, total = 0, 0
        pred = np.array([])
        label = np.array([])

        for task_id in range(self._cur_task + 1):
            if self._dataset != "MedMinist":
                cur_classes = data_manager.get_task_size(task_id)
                total_classes += cur_classes
                test_dataset = data_manager.get_dataset(np.arange(known_classes, total_classes), source='test', 
                                                    mode='test')
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                self._networks[task_id].eval()
                
                cur_correct, cur_total = 0, 0
                cur_pred = np.array([])
                cur_label = np.array([])
                for i, (_, inputs, targets) in enumerate(test_loader):
                    inputs = inputs.to(self._device)
                    with torch.no_grad():
                        outputs = self._networks[task_id](inputs)['logits']
                    predicts = torch.max(outputs, dim=1)[1]
                    
                    cur_correct += (predicts.cpu() == (targets - known_classes)).sum()
                    cur_total += len(targets)

                    cur_pred = np.concatenate([cur_pred, predicts.cpu().numpy()])
                    cur_label = np.concatenate([cur_label, targets.numpy() - known_classes])
                
                cur_acc = np.around(tensor2numpy(cur_correct)*100 / cur_total, decimals=2)
                cur_mcr = self.calculate_mean_class_recall(cur_pred, cur_label)
                self._acc_of_every_task[task_id].append(cur_acc)
                self._mcr_of_every_task[task_id].append(cur_mcr)               
            else:
                data_flag = self._dataset_order[task_id]
                info = INFO[data_flag]
                cur_classes = len(info['label'])
                total_classes += cur_classes
                DataClass = getattr(mymedmnist, info['python_class'])

                test_trsf = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                ])

                test_dataset = DataClass(split='test', transform=test_trsf, download=False, as_rgb=True)
                # encapsulate data into dataloader form
                test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
                self._networks[task_id].eval()

                cur_correct, cur_total = 0, 0
                cur_pred = np.array([])
                cur_label = np.array([])
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.to(self._device)
                    targets = targets.squeeze()
                    with torch.no_grad():
                        outputs = self._networks[task_id](inputs)['logits']
                    predicts = torch.max(outputs, dim=1)[1]
                    
                    cur_correct += (predicts.cpu() == targets).sum()
                    cur_total += len(targets)

                    cur_pred = np.concatenate([cur_pred, predicts.cpu().numpy()])
                    cur_label = np.concatenate([cur_label, targets.numpy()])
                
                cur_acc = np.around(tensor2numpy(cur_correct)*100 / cur_total, decimals=2)
                cur_mcr = self.calculate_mean_class_recall(cur_pred, cur_label)
                self._acc_of_every_task[task_id].append(cur_acc)
                self._mcr_of_every_task[task_id].append(cur_mcr) 
                
            correct += cur_correct
            total += cur_total
            pred = np.concatenate([pred, cur_pred + known_classes])
            label = np.concatenate([label, cur_label + known_classes])
            known_classes = total_classes 
            
        acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        mcr = self.calculate_mean_class_recall(pred, label)
        self._acc_of_all_task.append(acc)
        self._mcr_of_all_task.append(mcr)

        logging.info(50*"-")
        logging.info("log acc and mcr")
        logging.info(50*"-")

        recall = self.calculate_class_recall(pred, label)
        logging.info("recall of every class is {}".format(recall))
        
        logging.info("task acc of every task is {}".format(self._acc_of_every_task))
        logging.info("task acc of all task is {}".format(self._acc_of_all_task))
        logging.info("task mcr of every task is {}".format(self._mcr_of_every_task))
        logging.info("task mcr of all task is {}".format(self._mcr_of_all_task))