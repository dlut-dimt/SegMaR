import os, argparse
import torch
from utils.dataloader import test_dataset_distort
from attsampler_th import AttSampler
import imageio
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')

for _data_name in ['COD10K']: #'CAMO','CHAMELEON','COD10K'
    data_path = './COD10K/TestDataset/{}/'.format(_data_name)
    save_path = './Restort/'.format(_data_name)
    opt = parser.parse_args()
    os.makedirs(save_path, exist_ok=True)
    # put your image root, groundtruth root, prediction root and dilation map (pre stage) root respectively.
    image_root = './Imgs/'.format(data_path)
    gt_root = './GT/'.format(data_path)
    gte_root = './Prediction/'.format(data_path)
    gtl_root = './Dilation/'.format((data_path))
    test_loader = test_dataset_distort(image_root, gt_root, gte_root, gtl_root)

    for i in range(test_loader.size):

        image, gt, gte, gtl, name, name2 = test_loader.load_data()
        gt = gt.cuda()
        gte = gte.cuda()
        image = image.cuda()
        gtl = gtl.cuda()

        map_s = gtl
        map_sx = torch.unsqueeze(torch.max(map_s, 3)[0], dim=3)
        map_sx = torch.squeeze(map_sx, dim=1)
        map_sy = torch.unsqueeze(torch.max(map_s, 2)[0], dim=3)
        map_sy = torch.squeeze(map_sy, dim=1)
        sum_sx = torch.sum(map_sx, dim=(1, 2), keepdim=True)
        sum_sy = torch.sum(map_sy, dim=(1, 2), keepdim=True)
        map_sx /= sum_sx
        map_sy /= sum_sy

        res, grid = AttSampler(scale=1, dense=2)(gt, map_sx, map_sy)

        data_pred = gte
        ###########################restore#####################
        x_index = grid[0, 1, :, 0]  # 400
        y_index = grid[0, :, 1, 1]  # 300
        new_data_size = tuple(data_pred.shape[1:4])
        new_data = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                               device=gt.device)

        new_data_final = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                     device=gt.device)

        x_index = (x_index + 1) * new_data_size[2] / 2
        y_index = (y_index + 1) * new_data_size[1] / 2

        xl = 0
        grid_l = x_index[0]
        data_l = data_pred[:, :, :, 0]
        for num in range(1, len(x_index)):
            grid_r = x_index[num]
            xr = torch.ceil(grid_r) - 1
            xr = xr.int()
            data_r = data_pred[:, :, :, num]
            for h in range(xl + 1, xr + 1):
                if h == grid_r:
                    new_data[:, :, h] = data_r
                else:
                    new_data[:, :, h] = ((h - grid_l) * data_r / (grid_r - grid_l)) + ((grid_r - h) * data_l / (grid_r - grid_l))

            xl = xr
            grid_l = grid_r
            data_l = data_r

        new_data[:, :, 0] = new_data[:, :, 1]
        try:
            for h in range(xr + 1, len(x_index)):
                new_data[:, :, h] = new_data[:, :, xr]
        except:
            print('h', h)
            print('xr', xr)

        yl = 0
        grid1_l = y_index[0]
        data1_l = new_data[:, 0, :]
        for num in range(1, len(y_index)):
            grid1_r = y_index[num]
            yr = torch.ceil(grid1_r) - 1
            yr = yr.int()
            data1_r = new_data[:, num, :]
            for h in range(yl + 1, yr + 1):
                if h == grid1_r:
                    new_data_final[:, h, :] = data1_r
                else:
                    new_data_final[:, h, :] = ((h - grid1_l) * data1_r / (grid1_r - grid1_l)) + (
                            (grid1_r - h) * data1_l / (grid1_r - grid1_l))
            yl = yr
            grid1_l = grid1_r
            data1_l = data1_r
        new_data_final[:, 0, :] = new_data_final[:, 1, :]
        try:
            for h in range(yr + 1, len(y_index)):
                new_data_final[:, h, :] = new_data_final[:, yr, :]
        except:
            print('h', h)
            print('yr', yr)
        res = torch.unsqueeze(new_data_final, dim=1)

        res = res.squeeze(0)
        res = res.transpose(0, 2).contiguous()
        res = res.transpose(0, 1).contiguous()

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path + name, res)