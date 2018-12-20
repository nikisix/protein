#!/usr/bin/env python
# coding: utf-8

# # TODO, Notes, Questions

# In[1]:


# !pip install visdom


# In[2]:


import copy
import itertools
import time
from pathlib import Path
import os

from multiprocessing.dummy import Pool
import numpy as np
import pandas as pd
from PIL import Image
import PIL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import math
import torchvision
from torchvision import transforms
import tqdm

# from imgaug import augmenters as iaa

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib as mpl
mpl_params = {
    'figure.figsize': (10, 5),
    'figure.dpi': 300,
}
from matplotlib import pyplot as plt
mpl.rcParams.update(mpl_params)

import seaborn as sns
sns.set()

import visdom

import warnings
warnings.filterwarnings('ignore')

#pd.set_option('display.float_format', lambda x: '%.3f' % x)


# # Configuration

# In[3]:


# Human Protein Atlas Competetion Dataset
DATA_DIR = Path('../input/human-protein-atlas-image-classification/')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'sample_submission.csv')

# Preprocessed HPA Dataset
#    train: {0-31}-processed.npz 
#    test: {0-1}-test-processed.npz
#
#    1. 4 images to 1 RGB
#    2. resize to (299, 299)
#    3. [0, 255] to [0, 1]
#    4. apply standard normalization
PREPROCESS_DIR = Path('../input/hpa-images-processed/')
HPA_PREPROC_DIR = Path('../input/hpa_processed/')
# HPA_PREPROC_DIR = Path('../input/hpa-23142-drop-0-2-7-21-23-25/')
# hpa_df = pd.read_csv(HPA_PREPROC_DIR / 'hpa_select.csv')

HPA_DIR = Path('../input/HPAv18/')
hpa_df = pd.read_csv('../HPAv18RBGY_wodpl.csv')

# Saved model to continue training or tweak thresholds for submission
#    Can use an uploaded "dataset" called bestmodel:
#    ../input/bestmodel/best_model_195.pth
#    Or use previous kernel output:
#    ../input/inceptionv3-attention-wip/model.pth
CHECKPOINT_PATH = Path('best_model33.pth')
print('Model path exists', CHECKPOINT_PATH.exists())

# Set these three to False to quick commit so updates can be branched
# to new kernels to run different parameters simultaneously
SUBMISSION_RUN = True
TRAIN = True
LOAD_CHECKPOINT = True 

N_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SIGMOID_THRESHOLD = 0.5
VISDOM_ENV_NAME = 'dw_resnet34'

ADAPTIVE_POOLING = True


# In[4]:


vis = visdom.Visdom(env=VISDOM_ENV_NAME, server='http://3.17.85.107')


# # Device

# In[5]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


# # Labels

# In[6]:


LABELS = {
    0: 'Nucleoplasm', 
    1: 'Nuclear membrane',   
    2: 'Nucleoli',   
    3: 'Nucleoli fibrillar center' ,  
    4: 'Nuclear speckles',
    5: 'Nuclear bodies',
    6: 'Endoplasmic reticulum',   
    7: 'Golgi apparatus',
    8: 'Peroxisomes',
    9: 'Endosomes',
    10: 'Lysosomes',
    11: 'Intermediate filaments',   
    12: 'Actin filaments',
    13: 'Focal adhesion sites',   
    14: 'Microtubules',
    15: 'Microtubule ends',   
    16: 'Cytokinetic bridge',   
    17: 'Mitotic spindle',
    18: 'Microtubule organizing center',  
    19: 'Centrosome',
    20: 'Lipid droplets',   
    21: 'Plasma membrane',   
    22: 'Cell junctions', 
    23: 'Mitochondria',
    24: 'Aggresome',
    25: 'Cytosol',
    26: 'Cytoplasmic bodies',   
    27: 'Rods & rings'
}

LABEL_NAMES = list(LABELS.values())


# # Dataset

# In[7]:


class ProteinDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, train=True, device='cpu', preproc=True):            
        self.df = df.copy()
        self.device = device
        self._dir = images_dir
        self.transform = transform
        self.p = Pool(1)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(range(len(LABELS)))])
        self.colors = ['red', 'green', 'blue', 'yellow']
        self.train = train
        self.preproc = preproc
        self.cache = {}
        self.hpa_cache = {}
#         self.cache = self.load_cache(PREPROCESS_DIR, self.train)
#         self.hpa_cache = self.load_cache(HPA_PREPROC_DIR, self.train)

        
    def load_cache(self, preproc_dir, train=True):
        cache = {}
        filter_ = 'processed' if train else 'test'
        for npz in [x for x in preproc_dir.iterdir() if filter_ in x.name]:
            batch_key = int(npz.name.partition('-')[0])
            batch = np.load(npz)
            cache[batch_key] = batch
        return cache

    def __len__(self):
        return len(self.df)
    
    def mp_load(self, path):
        pil_im = Image.open(path)
        return np.array(pil_im, np.uint8)
                                      
    def __getitem__(self, key):
        """
        returns:
            X (image):
                dtype: torch.float32
                size: (299, 299) 
                device: self.device
            y (one hot encoded labels, e.g. [0, 1, 0, ...]):
                dtype:torch.float32
                size: (28,)
                device: self.device
        """
        id_ = self.df.loc[key].Id
        is_additional_hpa = self.df.loc[key].get('hpa', False)

        if self.preproc:
            if is_additional_hpa:
                preproc_dir = HPA_PREPROC_DIR
                cache = self.hpa_cache
                batch_size = 1022  # 798
            else:
                preproc_dir = PREPROCESS_DIR
                cache = self.cache
                batch_size = 971 if self.train else 5851
            batch_key = self.df.loc[key].original_index
            batch = batch_key // batch_size
            npz_key = f'arr_{batch_key - (batch_size * batch)}'
#             if batch not in cache:
#                 # Clear the dict cache is pointing to
#                 for k in list(cache.keys()):
#                     del cache[k]
#                 filename = f"{batch}-{'' if self.train else 'test-'}processed.npz"
#                 cache[batch] = np.load(preproc_dir / filename)
            X = cache[batch][npz_key]
        else:
            if is_additional_hpa:
                rgb = np.array(Image.open(HPA_DIR / f'{id_}.png'), np.uint8)
            else:
                image_paths = [self._dir / f'{id_}_{c}.png' for c in self.colors]
                r, g, b, y = self.p.map(self.mp_load, image_paths)
                rgb = np.stack([
                    r // 2 + y // 2,
                    g // 2 + y // 2,
                    b // 2
                ], axis=2)
            X = self.transform(rgb)

        y = []
        if 'Target' in self.df:
            y = list(map(int, self.df.iloc[key].Target.split(' ')))
            y = self.mlb.transform([y]).squeeze()  # TODO: This is weird.
        
        if self.train:
            X = self.dihedral(X)  # Could do on GPU if slow
        return X, y

    def dihedral(self, x):
        ''' Expected input shape is (channels, rows, cols) '''
        choice = np.random.randint(8)
        if choice == 0:  # no rotation
            x = np.rot90(x, k=0, axes=(1, 2)).copy()
        elif choice == 1:  # 90
            x = np.rot90(x, k=1, axes=(1, 2)).copy()
        elif choice == 2:  # 180
            x = np.rot90(x, k=2, axes=(1, 2)).copy()
        elif choice == 3:  # 270
            x = np.rot90(x, k=3, axes=(1, 2)).copy()
        elif choice == 4:  # no rotation mirror
            x = np.rot90(x, k=0, axes=(1, 2)).copy()
            x = np.flip(x, axis=2).copy()
        elif choice == 5:  # 90 mirror
            x = np.rot90(x, k=1, axes=(1, 2)).copy()
            x = np.flip(x, axis=1).copy()
        elif choice == 6:  # 180 mirror
            x = np.rot90(x, k=2, axes=(1, 2)).copy()
            x = np.flip(x, axis=2).copy()
        elif choice == 7:  # 270 mirror
            x = np.rot90(x, k=3, axes=(1, 2)).copy()
            x = np.flip(x, axis=1).copy()
        return x


# # DataLoaders

# In[8]:


len(hpa_df)


# In[9]:


aug_labels = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 27]


# In[10]:


selection_df = hpa_df[hpa_df.Target.apply(lambda x: bool(set(aug_labels).intersection(list(map(int, x.split())))))]


# In[11]:


len(selection_df)


# In[12]:


train_split, val_split = train_test_split(train_df, test_size=0.20, random_state=0)


# In[13]:


selection_df['hpa'] = True
train_split['hpa'] = False


# In[14]:


len(train_split)


# In[15]:


train_split = train_split.append(selection_df)


# In[16]:


len(train_split)


# In[17]:


def over_under_sampler(targets):
#     n_samples = {
#         0: 1, 25: 1,
#         21: 2, 23: 2,
#         2: 3, 4: 3, 5: 3, 7: 3,
#         1: 4, 3: 4, 6: 4, 11: 4, 14: 4, 18: 4, 19: 4, 22: 4,
#         12: 6, 13: 6, 16: 6, 17: 6, 24: 6, 26: 6,
#         8: 32, 9: 32, 10: 32, 15: 32, 20: 32, 27: 32, 
#     }
    n_samples = {
        0: 1, 25: 1,
        21: 1, 23: 1,
        2: 1, 4: 1, 5: 1, 7: 1,
        1: 1, 3: 1, 6: 1, 11: 1, 14: 1, 18: 1, 19: 1, 22: 1,
        12: 6, 13: 6, 16: 6,
        24: 16, 26: 16,
        8: 32, 9: 32, 10: 32, 15: 32, 20: 32, 27: 32, 17: 16,
    }
    multipliers = [n_samples[int(t)] for t in targets.split()]
    if targets == '0' or targets == '25 0' or targets == '25':
        return np.random.choice([0, 1], p=[.95, .05])
    elif '17' in targets:
        return 16
    elif 32 in multipliers:
        return 32
    else:
        return min(multipliers)


# In[18]:


train_split['original_index'] = train_split.index
train_split = train_split.reset_index(drop=True)
val_split['original_index'] = val_split.index
val_split = val_split.reset_index(drop=True)
test_df['original_index'] = test_df.index
test_df = test_df.reset_index(drop=True)


# In[19]:


train_split['oversample'] = train_split.Target.apply(over_under_sampler)


# In[20]:


train_split.head()


# In[21]:


train_split = train_split.loc[train_split.index.repeat(train_split.oversample)]
del train_split['oversample']


# In[22]:


len(train_split)


# In[23]:


train_split = train_split.reset_index(drop=True)


# In[24]:


train_split


# In[25]:


val_split.head()


# In[26]:


def plot_labels(df, ax):
    labels, counts = np.unique(list(map(int, itertools.chain(*df.Target.str.split()))), return_counts=True)  
    pd.DataFrame(counts, labels).plot(kind='bar', ax=ax)
    ax.set_ylim([0, 30000])


# In[27]:


#fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=200)
# plot_labels(train_df, ax1)
# plot_labels(train_split, ax2)


# In[28]:


transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((299, 299)),  # (299, 299) InceptionV3 input
    transforms.ToTensor(),  # To Tensor dtype and convert [0, 255] uint8 to [0, 1] float
    transforms.Normalize(  # Standard image normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

train_ds = ProteinDataset(
    train_split,
    images_dir=TRAIN_DIR,
    transform=transform,
    train=True,
    device=device,
    preproc=False,

)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)

val_ds = ProteinDataset(
    val_split,
    images_dir=TRAIN_DIR,
    transform=transform,
    train=True,
    device=device,
    preproc=False,
)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

dataloaders = {'train': train_dl, 'val': val_dl}

test_ds = ProteinDataset(
    test_df,
    images_dir=TEST_DIR,
    transform=transform,
    train=False,
    device=device,
    preproc=False,
)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)


# # Profile DataLoader

# In[29]:


#Time getting batch of images
# %time x = next(iter(train_dl))


# # Choose Layers To Train

# In[30]:


# for name, param in model.named_parameters():
#     print(name, param.requires_grad)


# # Get Model

# In[31]:


class Inception3Adaptive(torchvision.models.Inception3):
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = torch.cat([F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], 1)
        # 1 x 1 x 4096
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 4096
        x = x.view(x.size(0), -1)
        # 4096
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


# In[32]:


class AdamW(Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

        return loss


# In[60]:


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# In[61]:


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# In[62]:


class MyResNet(torchvision.models.ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.fc(x)

        return x


# In[70]:


# Download pretrained InceptionV3. transform_input is image normalization, which
# we have already done
model = torchvision.models.resnet34(pretrained=True)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.fc = nn.Linear(512, 28)

# if ADAPTIVE_POOLING:
#     pretrained = torchvision.models.inception_v3(pretrained=True, transform_input=False)
#     model = Inception3Adaptive(transform_input=False)
#     model.load_state_dict(pretrained.state_dict())
#     model.fc = nn.Linear(4096, 28)
#     model.aux_logits = False
# else:
#     model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
#     model.fc = nn.Linear(2048, 28)
#     model.aux_logits = False

# Replace 1000 output layer with 28 output layer for labels

sigmoid = nn.Sigmoid()

start_epoch = 0
if LOAD_CHECKPOINT:
    checkpoint = torch.load(CHECKPOINT_PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    # TODO: Make save function
    torch.save(checkpoint, f'loaded_checkpoint_model_{start_epoch}.pth')
    
# Freeze all layers
for name, param in model.named_parameters():
    param.requires_grad = False

if TRAIN:    
    for name, param in model.named_parameters():
        param.requires_grad = True 

    criterion = nn.BCEWithLogitsLoss()
    
    beginning = [{'params': p, 'lr': 1e-5} for n, p in list(model.named_parameters())[:3]]
    one = [{'params': p, 'lr': 1e-4} for n, p in model.named_parameters() if n.startswith('layer1')]
    two = [{'params': p, 'lr': 1e-3} for n, p in model.named_parameters() if n.startswith('layer2')]
    three = [{'params': p, 'lr': 1e-3} for n, p in model.named_parameters() if n.startswith('layer3')]
    four = [{'params': p, 'lr': 1e-2} for n, p in model.named_parameters() if n.startswith('layer4')]
    fc = [{'params': p, 'lr': 1e-2} for n, p in model.named_parameters() if n.startswith('fc')]

#     first = [{'params': p, 'lr': 1e-4} for n, p in model.named_parameters() if n.startswith('Conv2d')]
#     middle = [{'params': p, 'lr': 1e-3} for n, p in model.named_parameters() if n[:7] in ['Mixed_5', 'Mixed_6', 'Mixed_7', 'AuxLogi']]
#     last = [{'params': p, 'lr': 1e-2} for n, p in model.named_parameters() if n.startswith('fc')]
#     optim_params = first + middle + last
    optim_params = beginning + one + two + three + four + fc
    optimizer = AdamW(params=optim_params, lr=LEARNING_RATE, weight_decay=1e-5)
    if LOAD_CHECKPOINT:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

model.to(device)

# # Util

# In[71]:


class RunningStats():
    def __init__(self):
        self.reset()

    def reset(self):
        self.latest = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 0

    def update(self, val, n=1):
        self.latest = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min
        
class TrainingStats():
    def __init__(self):
        self.train_loss = RunningStats()
        self.train_f1 = RunningStats()
        self.val_loss = RunningStats()
        self.val_f1 = RunningStats()
        
    def plot(self):
        pass  # TODO


# 
# # Profile DataLoader / Estimate Train Time

# In[72]:


# # train_ds.transform = resize_transform(128)
# load_stat = RunningStats()
# proc_stat = RunningStats()
# test_size, test_iter = 16, 5
# test_batch_size_dl = DataLoader(train_ds, batch_size=test_size, shuffle=True, pin_memory=True)
# t1 = time.time()
# model.train()
# optimizer.zero_grad()
# test_dl_iter = iter(test_batch_size_dl)
# for _ in range(test_iter):
#     with torch.set_grad_enabled(True):
#         load_t1 = time.time()
#         X, y = next(test_dl_iter)
#         load_stat.update(time.time() - load_t1)
#         X = torch.as_tensor(X, dtype=torch.float32, device=device).cuda(non_blocking=True)
#         y = torch.as_tensor(y, dtype=torch.float32, device=device).cuda(non_blocking=True) if len(y) > 0 else y
#         proc_t1 = time.time()
#         y_ = model(X)
#         proc_stat.update(time.time() - proc_t1)
#         loss = criterion(y_, y)
#         loss.backward()
#         optimizer.step()
# batch_time = (time.time() - t1) / test_iter
# epoch_time = len(train_ds) / test_size * batch_time / 60
# train_time = epoch_time * N_EPOCHS / 60
# print(f'load time {load_stat.avg}')
# print(f'proc time {proc_stat.avg}')
# print(f'avg batch time: {batch_time:0.3f} s\nepoch est. {epoch_time:0.1f} m\ntrain est. {train_time:0.1f} h')


# ```
# load time 4.925958514213562
# proc time 0.05883393287658691
# avg batch time: 6.014 s
# epoch est. 149.0 m
# train est. 74.5 h
# ```

# ```
# load time 1.2517853021621703
# proc time 0.0170518159866333
# avg batch time: 2.326 s
# epoch est. 57.6 m
# train est. 28.8 h
# ```

# # Train

# In[76]:


def train(dataloaders, model, criterion, optimizer, sigmoid_thresh, n_epochs):
    start_ts = time.time()
    best_f1 = 0
    for epoch in range(n_epochs):
        stats = TrainingStats()
        total_epochs = start_epoch + epoch + 1
        print(f'Epoch {epoch + 1}/{n_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for X, y in tqdm.tqdm(dataloaders[phase]):
                X = torch.as_tensor(X, dtype=torch.float32, device=device).cuda(non_blocking=True)
                y = torch.as_tensor(y, dtype=torch.float32, device=device).cuda(non_blocking=True) if len(y) > 0 else y
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_ = model(X)
                    loss = criterion(y_, y)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                if phase == 'train':
                    stats.train_loss.update(loss.item(), n=X.shape[0])
                    f1 = f1_score(y.cpu(), sigmoid(y_.to('cpu')) > sigmoid_thresh, average='macro')
                    stats.train_f1.update(f1, n=X.shape[0])
                else:
                    stats.val_loss.update(loss.item(), n=X.shape[0])
                    f1, best_thresh = 0, 0
                    for thresh in np.linspace(.1, .9, 100):
                        score = f1_score(y.cpu(), sigmoid(y_.to('cpu')) > thresh, average='macro')
                        if score > f1:
                            f1, best_thresh = score, thresh
                    stats.val_f1.update(f1, n=X.shape[0])

        vis.line([stats.train_loss.avg], [total_epochs], update='append', opts={'title': 'Train Loss'}, win='Train Loss')
        vis.line([stats.train_f1.avg], [total_epochs], update='append', opts={'title': 'Train F1'}, win='Train F1')
        print(f'(train) loss: {stats.train_loss.avg} f1: {stats.train_f1.avg}')
        vis.line([stats.val_loss.avg], [total_epochs], update='append', opts={'title': 'Val Loss'}, win='Val Loss')
        vis.line([stats.val_f1.avg], [total_epochs], update='append', opts={'title': 'Val F1'}, win='Val F1')
        print(f'(val) loss: {stats.val_loss.avg} f1: {stats.val_f1.avg} thresh: {best_thresh}')
            
        if stats.val_f1.avg > best_f1 or epoch + 1 == n_epochs:
            best_f1 = stats.val_f1.avg
            save_state = dict(
                epoch=total_epochs,
                state_dict=model.state_dict(),
                f1=stats.val_f1.avg,
                loss=stats.val_loss.avg,
                optimizer=optimizer.state_dict(),
            )
            torch.save(save_state, f'best_model{total_epochs}.pth')


# In[77]:


def resize_transform(size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),  # (299, 299) InceptionV3 input
        transforms.ToTensor(),  # To Tensor dtype and convert [0, 255] uint8 to [0, 1] float
        transforms.Normalize(  # Standard image normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# In[ ]:


if TRAIN:
    train(dataloaders, model, criterion, optimizer, sigmoid_thresh=SIGMOID_THRESHOLD, n_epochs=N_EPOCHS)


# In[ ]:


# train(dataloaders, model, criterion, optimizer, sigmoid_thresh=.2, n_epochs=15)


# In[ ]:


# train128 = DataLoader(train_ds, batch_size=128, shuffle=True)
# val128 = DataLoader(val_ds, batch_size=128)
# dataloaders128 = {'train': train128, 'val': val128}
# for dl in dataloaders128.values():
#     dl.transforms = resize_transform((128, 128))

# t1 = time.time()
# train(dataloaders128, model, criterion, optimizer, sigmoid_thresh=.2, n_epochs=1)
# print(time.time() - t1)


# In[ ]:


# train256 = DataLoader(train_ds, batch_size=256, shuffle=True)
# val256 = DataLoader(val_ds, batch_size=256)
# dataloaders256 = {'train': train256, 'val': val256}
# for dl in dataloaders256.values():
#     dl.transforms = resize_transform((64, 64))

# t1 = time.time()
# train(dataloaders256, model, criterion, optimizer, sigmoid_thresh=.2, n_epochs=1)
# print(time.time() - t1)


# In[ ]:


# train512 = DataLoader(train_ds, batch_size=512, shuffle=True)
# val512 = DataLoader(val_ds, batch_size=512)
# dataloaders512 = {'train': train512, 'val': val512}
# for dl in dataloaders512.values():
#     dl.transforms = resize_transform((16, 16))

# t1 = time.time()
# train(dataloaders512, model, criterion, optimizer, sigmoid_thresh=.2, n_epochs=1)
# print(time.time() - t1)


# In[ ]:


# if SUBMISSION_RUN:
#     start_ts = time.time()
#     model.eval()

#     y_predictions = []
#     with torch.no_grad():
#         for X, y in test_dl:
#             X = torch.as_tensor(X, dtype=torch.float32, device=device).cuda(non_blocking=True)
#             y_ = model(X)
#             y_ = sigmoid(y_)
#             y_ = y_.to('cpu').numpy()
#             y_predictions.extend(y_)

#     y_predictions = np.stack(y_predictions)
#     print(f'Total eval time: {time.time() - start_ts:0.3f}s')


# In[50]:


# if SUBMISSION_RUN:
#     start_ts = time.time()
#     model.eval()

#     y_predictions = []
#     n_tta = 4
#     with torch.no_grad():
#         for i in range(len(test_ds)):
#             if i % 256 == 0:
#                 print(i, '/', len(test_ds))
#             y_ = []
#             for _ in range(n_tta):
#                 X, y = test_ds[i]
#                 X = torch.as_tensor(X, dtype=torch.float32, device=device).cuda(non_blocking=True)[None, :, :, :]
#                 y_.append(model(X))
#             y_ = torch.stack(y_).mean(0)
#             y_ = sigmoid(y_)
#             y_ = y_.to('cpu').numpy()
#             y_predictions.extend(y_)

#     y_predictions = np.stack(y_predictions)
#     print(f'Total eval time: {time.time() - start_ts:0.3f}s')


# In[43]:


# if SUBMISSION_RUN:
#     start_ts = time.time()
#     model.eval()

#     y_1 = []
#     y_2 = []
#     y_3 = []
#     y_4 = []

#     y_truth = []
#     n_tta = 1
#     with torch.no_grad():
#         for i in tqdm.tqdm_notebook(range(len(val_ds))):
#             if i % 256 == 0:
#                 print(i, '/', len(val_ds))
#             y_ = []
#             for _ in range(n_tta):
#                 X, y = val_ds[i]
#                 X = torch.as_tensor(X, dtype=torch.float32, device=device).cuda(non_blocking=True)[None, :, :, :]
#                 y_.append(model(X))
#             y_1.extend(sigmoid(y_[0]).to('cpu').numpy())
# #             y_2.extend(sigmoid(y_[1]).to('cpu').numpy())
# #             y_3.extend(sigmoid(y_[2]).to('cpu').numpy())
# #             y_4.extend(sigmoid(y_[3]).to('cpu').numpy())
#             y_truth.extend(y)

#     y_1 = np.stack(y_1)
# #     y_2 = np.stack(y_2)
# #     y_3 = np.stack(y_3)
# #     y_4 = np.stack(y_4)
#     y_truth = np.stack(y_truth)
#     print(f'Total eval time: {time.time() - start_ts:0.3f}s')


# In[47]:


# y2 = y_truth.reshape(5422, 28)


# In[66]:


# for i in range(28):
#     max_ = (0, 0)
#     for thresh in np.linspace(.1, .9, 100):
#         f1 = f1_score(y2[:, i], y_[:, i] > thresh, average='macro')
#         if f1 > max_[0]:
#             max_ = (f1, thresh)
#     print(i, thresh, f1)


# In[55]:


# for y_ in [y_1]:#, y_2, y_3, y_4]:
#     max_ = (0, 0)
#     for thresh in np.linspace(.1, .5, 100):
#         t = np.array([thresh] * 27 + [.])
# #         t[[8, 9, 10, 15, 17, 20, 24, 26, 27]] = thresh - .1
# #         t[[8, 9, 10, 27]] = thresh - .3
#         score = f1_score(y2, y_ > t, average='macro')
#         if score > max_[0]:
#             max_ = (score, thresh)
#             submit_thresh = thresh
#             submit_y_ = y_
#     print(max_)


# In[54]:


# if SUBMISSION_RUN:
#     submission = test_df[['Id', 'Predicted']].copy()
#     Predicted = []
#     for i, prediction in enumerate(test_ds.mlb.inverse_transform(y_ > submit_thresh)):
#         if len(prediction) == 0:
#             prediction = tuple([np.argmax(y_predictions[i])])
#         all_labels = []
#         for label in prediction:
#             all_labels.append(str(label))
#         Predicted.append(' '.join(all_labels))

#     submission['Predicted'] = Predicted

#     submission.to_csv('protein_classification.csv', index=False)


# In[55]:


# submission


# In[56]:


# np.save('y_sigmoid_predictions.npy', y_predictions)  # For offline work on setting thresholds


# In[63]:


if SUBMISSION_RUN:
    start_ts = time.time()
    model.eval()

    y_predictions = []
    with torch.no_grad():
        for X, _ in tqdm.tqdm(test_dl):
            X = torch.as_tensor(X, dtype=torch.float32, device=device).cuda(non_blocking=True)
            y_ = model(X)
            y_ = sigmoid(y_)
            y_ = y_.to('cpu').numpy()
            y_predictions.extend(y_)

    y_predictions = np.stack(y_predictions)
    print(f'Total eval time: {time.time() - start_ts:0.3f}s')


# In[68]:


if SUBMISSION_RUN:
    for t in [.2, .3, .4]:
        submission = test_df[['Id', 'Predicted']].copy()
        Predicted = []
        for i, prediction in enumerate(test_ds.mlb.inverse_transform(y_predictions > t)):
            if len(prediction) == 0:
                prediction = tuple([np.argmax(y_predictions[i])])
            all_labels = []
            for label in prediction:
                all_labels.append(str(label))
            Predicted.append(' '.join(all_labels))

        submission['Predicted'] = Predicted

        submission.to_csv(f'protein_classification{str(int(t * 10))}.csv', index=False)


# In[ ]:


np.save('y_sigmoid_predictions.npy', y_predictions)  # For offline work on setting thresholds

