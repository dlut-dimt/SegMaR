import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_SegMaR import Generator
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

dataset_path = './datasets/test/'

generator = Generator(channel=opt.feat_channel)
generator.load_state_dict(torch.load('./models/stage1/Model_50_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['COD10K']

for dataset in test_datasets:
    save_path = './results/' + 'COD10K' + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = './datasets/test/Imgs/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        generator_pred,_ = generator.forward(image)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)