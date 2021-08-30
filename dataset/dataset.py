import os, random, logging
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
   
    
class MyDataset(Dataset):
    def __init__(self, image_dir, img_list_path, transform, scale=1):
        self.image_dir = image_dir
        self.img_list_path = img_list_path
        self.scale = scale
        self.transform = transform
        self.dataset = []
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        with open(img_list_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            triple = [x.strip() for x in line.split(',')]
            self.dataset.append(triple)

        # logging.info(f'Creating dataset with {len(self.dataset)} examples')

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def preprocess(cls, pil_img, scale, transform=None):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        
        if transform!=None:
            # print("transform", transform)
            pil_img = transform(pil_img)

        img_nd = np.array(pil_img)
        # Remove alpha channel
        if img_nd.shape[-1]==4:
            img_nd = img_nd[:,:,0:3]

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, idx):
        triple = self.dataset[idx]
        # import pdb; pdb.set_trace()
        img_file = os.path.join(self.image_dir, triple[0]) 
        img_class = int(triple[1])
        mask_file = os.path.join(self.image_dir, triple[2]) 
        
        assert os.path.exists(img_file), \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file)  ## .resize(self.fixed_size)
        # mask = Image.open(mask_file[0])
        # img = Image.open(img_file[0])
        mask_torch = None
        if img_class==1:
            # print("img_class==1")
            assert os.path.exists(mask_file), \
                f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
            mask = Image.open(mask_file)   ##  .resize(self.fixed_size)
            
            assert img.size == mask.size, \
                f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
            
        
        seed = random.randint(0,2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.preprocess(img, self.scale, self.transform)
        img_torch = torch.from_numpy(img).type(torch.FloatTensor)
        
        if img_class==1:
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.preprocess(mask, self.scale, self.transform)
            mask_torch = torch.from_numpy(mask).type(torch.FloatTensor)
            # print("mask_torch==1")
        else:
            # print("mask shape:", img_torch.shape[1], img_torch.shape[2])
            mask_torch = torch.zeros(img_torch.shape[1], img_torch.shape[2]).type(torch.FloatTensor).unsqueeze(dim=0)

        return {
            'image': img_torch,
            'class': torch.tensor(img_class).unsqueeze(dim=0).type(torch.FloatTensor),
            'mask': mask_torch
        }