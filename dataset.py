import glob
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DistributedSampler, DataLoader
import random
import os
# import pdb


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        h='',
        shuffle=True,
        is_train=True
    ):
        if not os.path.exists(data_dir):
            raise Exception("{} not available!".format(data_dir))
        self.training_dirs = glob.glob(os.path.join(data_dir, '[0-9]*'))

        self.is_train = is_train
        if not self.is_train:
            self.num_samples = 40   # validation dataset sampling, since autoregressive decoding process costs much time
            self.samples = torch.randint(low=1, high=len(self.training_dirs), size=(self.num_samples,)).tolist()

        random.seed(1234)

        if shuffle:
            random.shuffle(self.training_dirs)

        self.img_nums_per_dir = 7

    def __getitem__(self, index):
        if self.is_train:
            img_dir = os.path.join(self.training_dirs[index], "im{}.png".format(random.randint(1, 7)))
        else:
            img_dir = os.path.join(self.training_dirs[self.samples[index]], "im{}.png".format(random.randint(1, 7)))
        image = Image.open(img_dir).convert('RGB')
        transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor()
        ])
        return transform(image)

    def __len__(self):
        return len(self.training_dirs) if self.is_train else self.num_samples

class KodacDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            h='',
    ):
        if not os.path.exists(data_dir):
            raise Exception("{} not available!".format(data_dir))
        self.image_dirs = glob.glob(os.path.join(data_dir, '*.png'))
        self.image_dirs = sorted(self.image_dirs)

    def __getitem__(self, index):
        img_dir = self.image_dirs[index]
        image = Image.open(img_dir).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_dirs)


def test_dataset():
    train_set_dir = "/data/xihuasheng/vimeo/video_train/vimeo_train"
    val_set_dir = "/data/xihuasheng/vimeo/vimeo_test"
    test_set_dir = "/data/liyao/kodac"

    # test validation set
    validset = Dataset(val_set_dir, h='')
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)
    for batch_idx, _ in enumerate(validation_loader):
        print(batch_idx)

    # test test set
    testset = KodacDataset(test_set_dir, h='')
    testset_loader = DataLoader(testset, num_workers=1, shuffle=False,
                                sampler=None,
                                batch_size=1,
                                pin_memory=True)
    for batch_idx, _ in enumerate(testset_loader):
        print(batch_idx)

    # test training set
    trainset = Dataset(train_set_dir, h='', shuffle=True)
    train_sampler = None
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=32,
                              pin_memory=True,
                              drop_last=True)
    for batch_idx, _ in enumerate(train_loader):
        print(batch_idx)

if __name__ == '__main__':
    test_dataset()