from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from copy import deepcopy
import torchvision.transforms as transforms


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = list(normal_class)
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes = [cls for cls in self.outlier_classes if cls not in self.normal_classes]
        # self.outlier_classes.remove(normal_class)
        print('Normal Classes: ', self.normal_classes)
        print('Outlier Classes: ', self.outlier_classes)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        
        # Make a list of transform to support multiclass outliers
        '''
        transform_dict = dict()
        for cls in range(0, 10):
            transform_dict[cls] = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[cls][0]],
                                                             [min_max[cls][1] - min_max[cls][0]])])
        '''
        

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class[0]][0]],
                                                             [min_max[normal_class[0]][1] - min_max[normal_class[0]][0]])])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        
        true_target_transform = transforms.Lambda(lambda x: self.normal_classes.index(x))
        
        train_set = MyMNIST(true_target_transform=true_target_transform, root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)

        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)

        self.train_set = Subset(train_set, train_idx_normal)
        
        self.test_set = MyMNIST(true_target_transform=true_target_transform, root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)


        print('Train set: %d examples' % len(self.train_set))
        print('Test set: %d examples' % len(self.test_set))
        
class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, true_target_transform=None, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)
        self.true_target_transform = true_target_transform

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
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
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.true_target_transform is not None and self.train:
            true_target = self.true_target_transform(true_target)
        else:
            true_target = true_target
            
        return img, target, index, true_target # only line changed
