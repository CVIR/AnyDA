import torch
from torchvision import transforms, datasets
#import params
from PIL import Image, ImageOps
import torch.utils.data as util_data
import utils.data_list
from utils.config import FLAGS
from utils.data_list import ImageList
import numpy as np
import numbers
import utils.utils_dino as utils_dino
from utils.transforms import Lighting, InputList, InputListdino
import random
from .randaugment import RandAugmentMC

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class TransformFixMatch(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.weak = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            ])
        self.strong = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.normal = transforms.Compose([    
                                        ResizeImage(256),
                                        transforms.RandomResizedCrop((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])  # ImageNet values
                                        ])

    def __call__(self, x):
        random_decider = random.uniform(0, 1)
        if random_decider < 0.5:
            #print("STRONG")
            return self.normalize(self.strong(x))
        else:
            #print("NORMAL")
            return self.normalize(self.weak(x))


####################################


def get_office_home(source_path, target_path, batch_size):

    data_transform_train = transforms.Compose([    
        ResizeImage(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # ImageNet values
        InputList(FLAGS.resolution_list),
    ])

    data_transform_train_aug = transforms.Compose([    
        TransformFixMatch(),
        InputList(FLAGS.resolution_list),
    ])
 
    resize_size=256
    crop_size=224
    start_center = (resize_size - crop_size - 1) / 2
 
    data_transform_test = transforms.Compose([
        ResizeImage(resize_size),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # ImageNet values
        InputList(FLAGS.resolution_list),
    ])

    if FLAGS.use_aug: 
        dsets_source = ImageList(open(source_path).readlines(), transform=data_transform_train_aug)
        dsets_target = ImageList(open(target_path).readlines(), transform=data_transform_train_aug)
    else:
        dsets_source = ImageList(open(source_path).readlines(), transform=data_transform_train)
        dsets_target = ImageList(open(target_path).readlines(), transform=data_transform_train)

    dsets_val = ImageList(open(target_path).readlines(), transform=data_transform_test)

    if FLAGS.is_cl_bln:
        source_train_loader = util_data.DataLoader(dsets_source, batch_size=FLAGS.s_bs, shuffle=False, num_workers=FLAGS.data_loader_workers,  drop_last=True, sampler=sampler)
    else:
        source_train_loader = util_data.DataLoader(dsets_source, batch_size=FLAGS.s_bs, shuffle=True, num_workers=FLAGS.data_loader_workers, drop_last=True)
    target_train_loader = util_data.DataLoader(dsets_target, batch_size=FLAGS.t_bs, shuffle=True, num_workers=FLAGS.data_loader_workers, drop_last=True)
    target_val_loader = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=False, num_workers=FLAGS.data_loader_workers, drop_last=True)

    print('Found {} Mini-batches for Office-Home (Source-Train) -- '.format(
        str(len(source_train_loader))))
    print('Found {} Mini-batches for Office-Home (Target-Train) -- '.format(
        str(len(target_train_loader))))
    print('Found {} Mini-batches for Office-Home (Target-Train) -- '.format(
        str(len(target_train_loader))))
    return source_train_loader, target_train_loader, target_val_loader
