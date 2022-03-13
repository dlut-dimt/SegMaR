import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy import misc
from lib.C2FNet import C2FNet
from utils.dataloader import test_dataset
import cv2

for _data_name in ['COD10K']: #'CAMO','CHAMELEON','COD10K'
    data_path = '/root/data/ysl/Dataset/COD10K/TestDataset/{}/'.format(_data_name)

    save_path = '/root/data/ysl/COD10K/TrainDataset/Edge_fix/'.format(_data_name)

    os.makedirs(save_path, exist_ok=True)
    image_root = '/root/data/ysl/COD10K/TrainDataset/Imgs/'.format(data_path)
    gt_root = '/root/data/ysl/COD10K/TrainDataset/GT/'.format(data_path)
    gte_root = '/root/data/ysl/COD10K/TrainDataset/Edge/'.format(data_path)
    gtl_root = '/root/data/ysl/COD10K/TrainDataset/fixation/'.format(data_path)

    test_loader = test_dataset(image_root, gt_root, gte_root, gtl_root)

    for i in range(test_loader.size):
        image, gt, gte, gtl, name, name2 = test_loader.load_data()
        kernelll = np.ones((25, 25), np.uint8)
        img = cv2.dilate(gte, kernelll, 2)
        kernel = (75, 75)
        img2 = cv2.GaussianBlur(img, kernel, 15)

        gtl = cv2.add(img2, gtl)
        gt = cv2.bitwise_and(gtl, gt)
        cv2.imwrite(save_path+name, gt)
        print(name)