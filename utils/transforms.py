"""
pytorch (0.3.1) miss some transforms, will be removed after official support.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as Func
import random
from torchvision import transforms, datasets
import utils.utils_dino as utils_dino

imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

class InputList(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        # assert img.size[0] == self.scales[0], 'image shape should be equal to max scale'
        # input_list = []
        # for i in range(len(self.scales)):
        #     input_list.append(F.resize(img, self.scales[i]))
         
        assert img.size()[1] == self.scales[0], 'image shape should be equal to max scale'
        input_list = []
        img = img[np.newaxis, :]
        for i in range(len(self.scales)):
            resized_img = Func.interpolate(img, (self.scales[i], self.scales[i]), mode='bilinear', align_corners=True)
            resized_img = torch.squeeze(resized_img)
            input_list.append(resized_img)

        return input_list

class InputListdino(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        # assert img.size[0] == self.scales[0], 'image shape should be equal to max scale'
        # input_list = []
        # for i in range(len(self.scales)):
        #     input_list.append(F.resize(img, self.scales[i]))
         
        assert img.size()[1] == self.scales[0], 'image shape should be equal to max scale'
        input_list = []
        img = img[np.newaxis, :]
        for i in range(len(self.scales)):
            resized_img = Func.interpolate(img, (self.scales[i], self.scales[i]), mode='bilinear', align_corners=True)
            resized_img = torch.squeeze(resized_img)
            input_list.append(resized_img)
        
        final_input_list = []
        for image in input_list:
            r_image = transforms.Compose([transforms.ToPILImage(),DataAugmentationDINO((0.4, 1.),(0.05, 0.4),0,),])(image) #8
            final_input_list.append(r_image)
        return final_input_list


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
 
        # first global crop
        self.global_transfo1 = transforms.Compose([
            #ResizeImage(256),
            #transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.GaussianBlur(1.0),
            normalize,
            #InputList(FLAGS.resolution_list),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            #ResizeImage(256),
            #transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.GaussianBlur(0.1),
            utils_dino.Solarization(0.2),
            normalize,
            #InputList(FLAGS.resolution_list),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            #ResizeImage(256),
            #transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.GaussianBlur(p=0.5),
            normalize,
            #InputList(FLAGS.resolution_list),
        ])
 
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class ListToTensor(object):
    def __call__(self, input_list):
        tensor_list = []
        for i in range(len(input_list)):
            pic = input_list[i]
            tensor_list.append(F.to_tensor(pic).detach())

        return tensor_list

class ListNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor_list):
        norm_list = []
        for i in range(len(tensor_list)):
            norm_list.append(F.normalize(tensor_list[i], self.mean, self.std, self.inplace))

        return norm_list