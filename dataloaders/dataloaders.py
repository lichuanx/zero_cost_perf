import os
import torch
import torchvision.datasets as dset 
import dataloaders.data_utils as utils

def define_dataloader(dataset='CIFAR10', data='data', batch_size=64):
    if dataset == 'CIFAR10':
        train_transform, valid_transform = utils._data_transforms_cifar10()
        train_data = dset.CIFAR10(root=data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=data, train=False, download=True, transform=valid_transform)
    elif dataset == 'CIFAR100':
        train_transform, valid_transform = utils._data_transforms_cifar100()
        train_data = dset.CIFAR100(root=data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=data, train=False, download=True, transform=valid_transform)
    elif dataset == 'SVHN':
        train_transform, valid_transform = utils._data_transforms_svhn()
        train_data = dset.SVHN(root=data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=data, split='test', download=True, transform=valid_transform)
    elif dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from .DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        valid_data = ImageNet16(root=os.path.join(data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

