from models.OneShotNet import MultiTaskNetOneShot
from dataset.dataset import MyDataset
from dataset.MyRandomResizeTransform import MyRandomResizeTransform
from torch.utils.data import DataLoader, random_split
from dataset.my_data_loader import MyDataLoader

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys, os, time, pickle, random
import logging
import argparse
    
def get_args():
    parser = argparse.ArgumentParser("Supernet")
    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--total-epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='init learning rate')
    parser.add_argument('--save', type=str, default='checkpoints', help='dir for saving checkpoint')
    parser.add_argument('--img-dir', type=str, default='../', help='dir for loading images')
    parser.add_argument('--train-list', type=str, default='../dataset/train.txt', help='path to training list')
    parser.add_argument('--valid-list', type=str, default='../dataset/valid.txt', help='path to validation list')
    
    parser.add_argument('--image-size', type=list, default=[192, 256, 320, 384, 448], help='image resolution list')

    args = parser.parse_args()
    return args

def dice(target, predictive, ep=1e-8):
    interp = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    dice = interp / union
    return dice

def random_code(code_len=18):
    code = []
    upsampling_layers_idx = [0,3,6,9,12,15]
    up_op_len = 2
    conv_op_len = 4
    for i in range(code_len):
        if i in upsampling_layers_idx:
            code.append(np.random.randint(up_op_len))
        else:
            code.append(np.random.randint(conv_op_len))
    return code

def train(model, device, loader, optimizer):
    _lambda = 1e-1
    model.train()
    train_cls_loss_all = []
    train_dice_all = []
    correct = 0
    sample_num_so_far = 0
    
    labels = [int(x[1]) for x in loader.dataset.dataset]
    class_count = torch.bincount(torch.tensor(labels))
    class_weighting = class_count.sum()/class_count
    
    for batch, pair in enumerate(loader):
        data, target_cls, target_mask = pair['image'].to(device), pair['class'].to(device), pair['mask'].to(device)
        
        optimizer.zero_grad()
        # forward
        code = random_code()
        pred_cls, pred_seg = model(data, code)
        
        # Classification loss
        total_loss = 0
        cls_bce_loss = F.binary_cross_entropy_with_logits(pred_cls, target_cls, class_weighting[target_cls.long()])
        train_cls_loss_all.append(cls_bce_loss.item())
        total_loss += _lambda * cls_bce_loss
        
        correct += ((torch.sigmoid(pred_cls)>0.5).type(torch.IntTensor).to(device) == target_cls).sum().item()
        sample_num_so_far += data.shape[0]
        
        if target_cls.sum()>0:
            idx_mask = (target_cls==1)
            # Segmentation loss
            seg_bce_loss = F.binary_cross_entropy_with_logits(pred_seg[idx_mask], target_mask[idx_mask])
            seg_dice = dice(target_mask[idx_mask], torch.sigmoid(pred_seg[idx_mask])) 
            seg_dice_loss = 1-seg_dice
            total_loss += seg_dice_loss    
            train_dice_all.append(seg_dice.item())
        
        total_loss.backward()
        optimizer.step()
        
        if batch % 100 ==0:
            logging.info("Train dice:{:.6f}, cls acc:{:.6f}, cls loss:{:.6f}, total loss:{:.6f}".format(np.mean(train_dice_all), correct/sample_num_so_far, np.mean(train_cls_loss_all), total_loss.item()))
                
    return np.mean(train_dice_all)


def valid(model, device, loader):
    model.eval()
    valid_dice_all = []
    n_val = len(loader)  # the number of batch
    correct = 0
    sample_num_so_far = 0
    
    with torch.no_grad():
        for batch, pair in enumerate(loader):
            data, target_cls, target_mask = pair['image'].to(device), pair['class'].to(device), pair['mask'].to(device)
            
            code = random_code()
            pred_cls, pred_seg = model(data, code)
            
            # Classification loss            
            correct += ((torch.sigmoid(pred_cls)>0.5).type(torch.IntTensor).to(device) == target_cls).sum().item()
            sample_num_so_far += data.shape[0]
        
            if target_cls.sum()>0:
                idx_mask = (target_cls==1)
                # Segmentation loss
                seg_bce_loss = F.binary_cross_entropy_with_logits(pred_seg[idx_mask], target_mask[idx_mask])
                seg_dice = dice(target_mask[idx_mask], torch.sigmoid(pred_seg)[idx_mask])                 
                valid_dice_all.append(seg_dice.item())
            
    valid_dice_mean = np.mean(valid_dice_all)
    logging.info("Valid dice:{:.6f}, cls acc:{:.6f}".format(valid_dice_mean, correct/sample_num_so_far))
    return valid_dice_mean
    
def valid_all(model, device, loader):
        model.eval()
        valid_dice_all = []
        n_val = len(loader)  # the number of batch
        correct = 0
        sample_num_so_far = 0
        pred_prob_all = []
        ground_truth = []
        code = [1,]*18
        with torch.no_grad():
            for batch, pair in enumerate(loader):
                data, target_cls, target_mask = pair['image'].to(device), pair['class'].to(device), pair['mask'].to(device)

                pred_cls, pred_seg = model(data, code)
                pred_prob_all.append(torch.sigmoid(pred_cls).cpu().numpy()[0, 0])
                ground_truth.append(target_cls.cpu().numpy()[0, 0])
                # Classification loss            
                correct += ((torch.sigmoid(pred_cls)>0.5).type(torch.IntTensor).to(device) == target_cls).sum().item()
                sample_num_so_far += data.shape[0]

                if target_cls.sum()>0:
                    idx_mask = (target_cls==1)
                    # Segmentation loss
                    seg_bce_loss = F.binary_cross_entropy_with_logits(pred_seg[idx_mask], target_mask[idx_mask])
                    seg_dice = dice(target_mask[idx_mask], torch.sigmoid(pred_seg)[idx_mask])                 
                    valid_dice_all.append(seg_dice.item())

        valid_dice_mean = np.mean(valid_dice_all)
        logging.info("Valid dice:{:.6f}, cls acc:{:.6f}".format(valid_dice_mean, correct/sample_num_so_far))
        return valid_dice_mean, correct/sample_num_so_far, pred_prob_all, ground_truth

def calc_metrics(pred_prob_all, ground_truth):
    idx_sp = (np.array(ground_truth)==0)
    y1 = (np.array(pred_prob_all)>0.5)[idx_sp]
    y2 = (np.array(ground_truth)==1)[idx_sp]
    sp = np.mean(y1==y2)
    
    idx_se = (np.array(ground_truth)==1)
    y1 = (np.array(pred_prob_all)>0.5)[idx_se]
    y2 = (np.array(ground_truth)==1)[idx_se]
    se = np.mean(y1==y2)
    
    acc = np.mean((np.array(pred_prob_all)>0.5)  == (np.array(ground_truth)==1))
    
    return se, sp, acc

def main():
    args = get_args()
    dir_checkpoint = args.save  # 'checkpoints'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
        
    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)

    fh = logging.FileHandler(os.path.join(dir_checkpoint, 'train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    # Record file path and content
    logger = logging.getLogger()
    filepath = os.path.abspath(__file__)
    logger.info(filepath)
    logger.info(args)
    
    # with open(filepath, "r") as f:
    #     logger.info(f.read())
        
    # Data Loader   
    train_list = args.train_list  
    valid_list = args.valid_list   
    logging.info("Load data from {}".format(train_list))
    logging.info("Load data from {}".format(valid_list))

    
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    image_size = args.image_size
    logger.info("Sampling image size:{}".format(image_size))
    MyRandomResizeTransform.IMAGE_SIZE_LIST = image_size.copy()
    MyRandomResizeTransform.ACTIVE_SIZE = max(image_size)
    resize_transforms = MyRandomResizeTransform(size=0)
    
    dataset = MyDataset(args.img_dir, train_list, resize_transforms)
    dataset_val = MyDataset(args.img_dir, valid_list, resize_transforms)
        
    logging.info("train len: {}, val len: {}".format(len(dataset), len(dataset_val)))
    n_epochs = args.total_epochs  
    
    device = "cuda:0"
    model = MultiTaskNetOneShot(n_classes=1, c_classes=1, pretrained_encoder=True)
    # logging.info(model)
    
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) 

    loader = MyDataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    loader_val = MyDataLoader(dataset_val, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
    
    best_dice = 0
    for epoch in range(n_epochs):
        logging.info("epoch:{}".format(epoch))
        training_history = []
        # learning rate decay
        decay_epochs = [int(n_epochs*0.5), int(n_epochs*0.75)]
        if epoch in decay_epochs:
            logging.info('Changing learning rate!')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        logging.info('learning rate:{}'.format(optimizer.param_groups[0]['lr']))
        
        train_dice = train(model, device, loader, optimizer)
        
        valid_dice, acc, pred_prob_all, ground_truth = valid_all(model, device, loader_val)
        se, sp, acc = calc_metrics(pred_prob_all, ground_truth)
        
        # Save checkpoint
        if valid_dice > best_dice:
            best_dice = valid_dice
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, 'model_{:.4f}.pth'.format(valid_dice)))
            logging.info("***Val dice improved at {}-th epoch".format(epoch))
            logging.info("se:{}, sp:{}, acc:{}".format(se, sp, acc))
        torch.save(model.state_dict(), os.path.join(dir_checkpoint, 'model.pth'))
        
    training_history.append([train_dice, valid_dice])
    
    history_file = os.path.join(dir_checkpoint, "training_history.pkl")
    with open(history_file, 'wb') as f:
        pickle.dump(training_history, f)

    
if __name__=="__main__":
    main()
