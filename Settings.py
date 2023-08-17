import os
import numpy as np
import copy as cp
import random as rd

seed = 1234
np.random.seed(seed)
rd.seed(seed)

import torch
import torch.nn as nn
import gc
import time
import math
from collections import OrderedDict
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from os.path import join
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from Comp_FIM.object import PMatKFAC, PMatEKFAC, PMatDiag, PMatDense
from Comp_FIM.metrics import FIM, FIM_MonteCarlo
from Comp_FIM.object.vector import random_pvector, PVector

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

## Configurations:
device = "cuda"
BaseRoot = "//"
ShakeRoot = "//"
Symbol = "//"


