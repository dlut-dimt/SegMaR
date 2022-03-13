import os, argparse
import torch
from utils.dataloader import test_dataset_distort
from attsampler_th import AttSampler
import imageio
parser = argparse.ArgumentParser()

for _data_name in ['COD10K']: #'CAMO','CHAMELEON','COD10K'
    data_path = './COD10K/TestDataset/{}/'.format(_data_name)
    save_path = './Distort/Imgs/'.format(_data_name)
    save_path2 = './Distort/GT/'.format(_data_name)
    save_path3 = './Distort/Dis/'.format(_data_name)
    opt = parser.parse_args()
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)
    os.makedirs(save_path3, exist_ok=True)
    # put your image root, groundtruth root, other mask root and dilation map (this stage) root respectively.
    image_root = './Original/Imgs/'.format(data_path)
    gt_root = './Original/GT/'.format(data_path)
    gte_root = './Original/Dis/'.format((data_path))
    gtl_root = './Original/Dilation/'.format((data_path))
    test_loader = test_dataset_distort(image_root, gt_root, gte_root, gtl_root)

    for i in range(test_loader.size):
        image, gt, gte, gtl, name, name2 = test_loader.load_data()
        gt = gt.cuda()
        gte = gte.cuda()
        image = image.cuda()
        gtl = gtl.cuda()

        #-------------attention-based(gtl) Sampler-----------------
        map_s = gtl
        map_sx = torch.unsqueeze(torch.max(map_s, 3)[0], dim=3)  # ([1, 256, 1])
        map_sx = torch.squeeze(map_sx, dim=1)
        map_sy = torch.unsqueeze(torch.max(map_s, 2)[0], dim=3)  # ([1, 256, 1])
        map_sy = torch.squeeze(map_sy, dim=1)
        sum_sx = torch.sum(map_sx, dim=(1, 2), keepdim=True)
        sum_sy = torch.sum(map_sy, dim=(1, 2), keepdim=True)
        map_sx /= sum_sx
        map_sy /= sum_sy

        res, grid = AttSampler(scale=1, dense=2)(image, map_sx, map_sy)
        resgt, grid = AttSampler(scale=1, dense=2)(gt, map_sx, map_sy)
        resgte, grid = AttSampler(scale=1, dense=2)(gte, map_sx, map_sy)
        x_index = grid[0, 1, :, 0]  # 400
        y_index = grid[0, :, 1, 1]  # 300

        x_index = (x_index + 1) * 275 / 2
        y_index = (y_index + 1) * 400 / 2

        for num in range(1, len(x_index)):
            grid_r = x_index[num]
            for num in range(1, len(y_index)):
                grid1_r = y_index[num]

        res = res.squeeze(0)
        res = res.transpose(0, 2).contiguous()
        res = res.transpose(0, 1).contiguous()
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name2, res)
        imageio.imsave(save_path + name, res)

        resgt = resgt.squeeze(0)
        resgt = resgt.transpose(0, 2).contiguous()
        resgt = resgt.transpose(0, 1).contiguous()
        resgt = resgt.sigmoid().data.cpu().numpy().squeeze()
        resgt = (resgt - resgt.min()) / (resgt.max() - resgt.min() + 1e-8)
        imageio.imsave(save_path2 + name, resgt)

        resgte = resgte.squeeze(0)
        resgte = resgte.transpose(0, 2).contiguous()
        resgte = resgte.transpose(0, 1).contiguous()
        resgte = resgte.sigmoid().data.cpu().numpy().squeeze()
        resgte = (resgte - resgte.min()) / (resgte.max() - resgte.min() + 1e-8)
        imageio.imsave(save_path3 + name, resgte)