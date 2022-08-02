import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist,Imagelists_VISDA_Test
from loaders.data_list import Imagelists_VISDA, return_classlist, Imagelists_VISDA_un

import pickle
from pdb import set_trace as breakpoint
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random

from PIL import Image
import copy
import pdb
from torch.utils.data.dataloader import default_collate

from loaders.data_list import make_dataset_fromlist

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

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def return_dataset_balance_self(args,test=False):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=3, m=10),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None


    train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    if test:
        target_dataset_test = Imagelists_VISDA_Test(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    else:
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root, transform=data_transforms['val'],transform2=data_transforms['self'])
    
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img) 

    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    nw = 12


    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
        num_workers=nw, shuffle=True, drop_last=True)
    labeled_target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=bs,
        num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=False)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=False)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*4, num_workers=nw, shuffle=True, drop_last=True)

    return source_loader, labeled_target_loader, target_loader_val, target_loader_test, target_loader_unl, class_list



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list
