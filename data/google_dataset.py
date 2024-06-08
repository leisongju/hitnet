import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import torch
import torchvision.transforms.functional as TF
import time

class GoogleDataset(Dataset):
    def __init__(self, data_dir, train=True, image_height=256, image_width=512):
        super(GoogleDataset, self).__init__()
        self.datapath = data_dir
        self.train = train
        self.image_height = image_height
        self.image_width = image_width
        self.file_paths = self._load_file_paths()
    
    def _load_file_paths(self):
        subdir = 'train' if self.train else 'test'
        pattern = os.path.join(self.datapath, subdir, "**", 'result_pd_left_*.png')
        file_paths = glob(pattern, recursive=True)
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def random_augmentation(self, under):
        # 确保under是一个Tensor，使用shape获取尺寸
        _, h, w = under.shape  # 假设under是C x H x W的Tensor
        w_start = max(0, w - self.image_width)
        h_start = max(0, h - self.image_height)
        
        random_w = torch.randint(low=0, high=w_start + 1, size=(1,)).item()
        random_h = torch.randint(low=0, high=h_start + 1, size=(1,)).item()
        
        return random_w, random_h

    def __getitem__(self, idx):
        l_img_path = self.file_paths[idx]
        r_img_path = l_img_path.replace("/raw_left_pd/", "/raw_right_pd/").replace("/result_pd_left_", "/result_pd_right_")
        
        # 读取左图和右图
        l_img = cv2.imread(l_img_path, -1)
        r_img = cv2.imread(r_img_path, -1)
        
        # 将图像归一化，并转换为Tensor
        l_img = torch.tensor(l_img / 16368.0).unsqueeze(0).float()
        r_img = torch.tensor(r_img / 16368.0).unsqueeze(0).float()

        # 随机裁剪
        l_start_w, l_start_h = self.random_augmentation(l_img)
        l_img_crop = l_img[:, l_start_h:l_start_h + self.image_height, l_start_w:l_start_w + self.image_width]
        r_img_crop = r_img[:, l_start_h:l_start_h + self.image_height, l_start_w:l_start_w + self.image_width]

        # 旋转图像
        # l_img_flip = l_img_crop.permute(0, 2, 1).flip(2)
        # r_img_flip = r_img_crop.permute(0, 2, 1).flip(2)

        return l_img_crop, r_img_crop

if __name__ == '__main__':
    datapath = '/mnt/datalsj/dual_pixel/data/google'
    dataset = GoogleDataset(datapath, train=True, image_height=256, image_width=512)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0) #, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    # 测试加载速度
    start_time = time.time()
    for i, (l_imgs, r_imgs) in enumerate(dataloader):
        if i == 0:
            print(f"Batch 0 - Left image shape: {l_imgs.shape}, Right image shape: {r_imgs.shape}")
        if i % 100 == 0:  # 测试每100个批次的时间
            end_time = time.time()
            print(f"加载100个批次的时间: {end_time - start_time:.2f} 秒")
            start_time = time.time()
