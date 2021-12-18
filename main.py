import glob
import os
import re

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import scipy
import scipy.ndimage
import scipy.io as io
import scipy.spatial as spatial
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary
from tensorboardX import SummayWriter

torch.backends.cudnn.benchmark=False#因为输入图片大小不确定，所以设置其为False，下文有详细介绍
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#使用gpu进行训练

#数据集展示
sample_index=6#样本编号
sample_root="/datasets/part_A_final/train_data"#样本根目录
img_path=os.path.join(sample_root,f"images/IMG_{sample_index}.jpg")#图片
gt_mat_path=os.path.join(sample_root,f"ground_truth/GT_IMG_{sample_index}.mat")#标签

img=plt.imread(img_path)
gt=io.loadmat(gt_mat_path)