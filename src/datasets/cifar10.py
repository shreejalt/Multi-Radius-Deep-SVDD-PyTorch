from copy import deepcopy
from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms


class CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        
        self.normal_classes = list(normal_class)
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes = [cls for cls in self.outlier_classes if cls not in self.normal_classes]
        # self.outlier_classes.remove(normal_class)
        print('Normal Classes: ', self.normal_classes)
        print('Outlier Classes: ', self.outlier_classes)
        print(self.normal_classes)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        # Make a list of transform to support multiclass outliers
        transform_dict = dict()
        '''
        for cls in range(0, 10):
            transform_dict[cls] = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[cls][0]],
                                                             [min_max[cls][1] - min_max[cls][0]])])
        '''
        
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class[0]][0]] * 3,
                                                             [min_max[normal_class[0]][1] - min_max[normal_class[0]][0]] * 3)])
        
        
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        true_target_transform = transforms.Lambda(lambda x: self.normal_classes.index(x))
        
        train_set = MyCIFAR10(true_target_transform=true_target_transform, root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCIFAR10(true_target_transform=true_target_transform, root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)

        print('Training set size: %d' % len(self.train_set))
        print('Test set size: %d' % len(self.test_set))
        

class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, true_target_transform=None, *args, **kwargs):
        self.true_target_transform = true_target_transform
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        true_target = deepcopy(target)
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.true_target_transform is not None and self.train:
            true_target = self.true_target_transform(true_target)
        else:
            true_target = true_target
        
        return img, target, index, true_target  # only line changed
