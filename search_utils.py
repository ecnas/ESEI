from models.OneShotNet import MultiTaskNetOneShot
from dataset.dataset import MyDataset
from dataset.MyRandomResizeTransform import MyRandomResizeTransform, MyDeterministicResizeTransform
from torch.utils.data import DataLoader, random_split
from dataset.my_data_loader import MyDataLoader

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys, os, time, pickle, random
import logging
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class DNN():
    def __init__(self, model_path, img_dir, valid_list, image_size, batch_size):
        self.device = "cuda:0"
        self.model = MultiTaskNetOneShot(n_classes=1, c_classes=1, pretrained_encoder=True)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.valid_list = valid_list
        self.image_size = image_size
        self.image_dir = img_dir
        self.batch_size = batch_size
    
    def dice(self, target, predictive, ep=1e-8):
        interp = 2 * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        dice = interp / union
        return dice

    def random_code(self, code_len=18):
        code = [random.choice(list(range(len(self.image_size))))]
        upsampling_layers_idx = [0,3,6,9,12,15]
        up_op_len = 2
        conv_op_len = 4
        for i in range(code_len):
            if i in upsampling_layers_idx:
                code.append(np.random.randint(up_op_len))
            else:
                code.append(np.random.randint(conv_op_len))
        return code

    def valid(self, model, device, loader, code):
        model.eval()
        valid_dice_all = []
        n_val = len(loader)  # the number of batch
        correct = 0
        sample_num_so_far = 0

        with torch.no_grad():
            for batch, pair in enumerate(loader):
                data, target_cls, target_mask = pair['image'].to(device), pair['class'].to(device), pair['mask'].to(device)

                # print(data.shape)
                pred_cls, pred_seg = model(data, code)

                # Classification loss            
                correct += ((torch.sigmoid(pred_cls)>0.5).type(torch.IntTensor).to(device) == target_cls).sum().item()
                sample_num_so_far += data.shape[0]

                if target_cls.sum()>0:
                    idx_mask = (target_cls==1)
                    # Segmentation loss
                    seg_bce_loss = F.binary_cross_entropy_with_logits(pred_seg[idx_mask], target_mask[idx_mask])
                    seg_dice = self.dice(target_mask[idx_mask], torch.sigmoid(pred_seg)[idx_mask])                 
                    valid_dice_all.append(seg_dice.item())

        valid_dice_mean = np.mean(valid_dice_all)
        logging.info("Valid dice:{:.6f}, cls acc:{:.6f}".format(valid_dice_mean, correct/sample_num_so_far))
        return valid_dice_mean, correct/sample_num_so_far

    def eval_solution(self, code):
        resolution = [self.image_size[code[0]],]*2
        val_transforms = MyDeterministicResizeTransform(size=resolution)        
        dataset_val = MyDataset(self.image_dir, self.valid_list, val_transforms)
        loader_val = DataLoader(dataset_val, batch_size=self.batch_size, num_workers=8, pin_memory=False, shuffle=False)
        logging.info('Resolution:{}, sample num:{}'.format(resolution, len(dataset_val)))
        valid_dice, acc = self.valid(self.model, self.device, loader_val, code[1:])
        del dataset_val, loader_val
        return valid_dice, acc
