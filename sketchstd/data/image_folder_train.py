"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
from pdb import set_trace as st
import itertools


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    #dir = dir.replace(dir.split("/")[-1] , "")
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    dir_ = dir
    sub_list = [os.path.join(dir_, o) for o in os.listdir(dir_) if os.path.isdir(os.path.join(dir_,o))]
    images = []
    for sub_dirr in sub_list:
        images.append ([os.path.join(sub_dirr, o) for o in os.listdir(sub_dirr) if os.path.isdir(os.path.join(sub_dirr,o)) and "." not in o] )
        #images.append([os.path.join(sub_dirr, file) for file in os.listdir(sub_dirr) if file.endswith('.png')])
    images = list(itertools.chain(*images))
    
    return images[:min(max_dataset_size, len(images))]
    """st()
    
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]"""


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
