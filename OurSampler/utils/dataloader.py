import os
from PIL import Image
import torchvision.transforms as transforms

#distort
class test_dataset_distort:
    def __init__(self, image_root, gt_root, gte_root, gtl_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gtes = [gte_root + f for f in os.listdir(gte_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gtls = [gtl_root + f for f in os.listdir(gtl_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gtes = sorted(self.gtes)
        self.gtls = sorted(self.gtls)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.gt_transform = transforms.ToTensor()
        self.gtl_transform = transforms.Compose([
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)  #restort and distort
        gte = self.binary_loader(self.gtes[self.index])
        gte = self.gt_transform(gte).unsqueeze(0)  # restort and distort
        gtl = self.binary_loader(self.gtls[self.index])
        gtl = self.gtl_transform(gtl).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        name2 = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        if name2.endswith('.jpg'):
            name2 = name2.split('.jpg')[0] + '.jpg'
        self.index += 1
        return image, gt, gte, gtl, name, name2

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')