import random
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda import amp
import pandas as pd
from config import *
import timm
import process
import model as mo


def set_seed(seed=42):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, optimizer, losses_dict):
    model.train()
    scaler = amp.GradScaler()
    losses_all, ce_all = 0, 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    # p = pbar[0]
    for _, (images, gt) in pbar:
        optimizer.zero_grad()

        images = images.to(device, dtype=torch.float)  # [b, c, w, h]
        gt = gt.to(device)

        with amp.autocast(enabled=True):
            y_preds = model(images)
            ce_loss = losses_dict["CELoss"](y_preds, gt.long())
            losses = ce_loss

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_all += losses.item() / images.shape[0]
        ce_all += ce_loss.item() / images.shape[0]

    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all, ce_all), flush=True)


@torch.no_grad()
def valid_one_epoch_ture(model, valid_loader):
    model.eval()
    df = pd.DataFrame( columns=['label','value'])
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, gt) in pbar:
        images = images.to(device, dtype=torch.float)  # [b, c, w, h]

        gt = gt.to(device)
        list_label = gt.cpu().numpy().tolist()
        y_preds = model(images)
        prob = torch.nn.functional.softmax(y_preds, dim=-1)[:, 1].detach().cpu().numpy().tolist()

        res_array = [list_label, prob]
        res_array = np.array(res_array).T
        est_df = pd.DataFrame(res_array, columns=['label', 'value'])
        est_df.label = est_df.label.astype('int')
        df = pd.concat([df,est_df],ignore_index=True)

    pred_untampers = df.query('label==0')
    pred_tampers = df.query('label==1')
    thres = np.percentile(pred_untampers.values[:, 1], np.arange(90, 100, 1))
    recall = np.mean(np.greater(pred_tampers.values[:, 1][:, np.newaxis], thres).mean(axis=0))
    print("recall: {:.2f}".format(recall * 100), flush=True)

    return recall * 100


@torch.no_grad()
def test_one_epoch(test_df, ckpt_paths, test_loader):
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids) in pbar:

        images = images.to(device, dtype=torch.float)  # [b, c, w, h]
        y_preds = 0
        i =1
        for sub_ckpt_path in ckpt_paths:
            model = mo.build_model(pretrain_flag=True)  # just dummy code
            # model.load_state_dict(torch.load(sub_ckpt_path))
            weights_dict = torch.load(sub_ckpt_path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)

            model.eval()
            y_pred = model(images)  # [b, c, w, h]

            # TTA
            if 1:
                flips = [[-1],[-2]]
                for f in flips:
                    images = torch.flip(images,f)
                    output = model(images)
                    y_pred += output
            y_pred /= len(flips) + 1
            y_preds += y_pred
            y_preds = y_preds / 4

            prob = torch.nn.functional.softmax(y_preds, dim=-1)[:, 1].detach().cpu().numpy()
            test_df.loc[test_df['img_name'].isin(ids), 'pred_prob'+str(i)] = prob
            i = i+1
            # test_df.loc[test_df['img_name'].isin(ids), 'pred_prob'] = prob

    return test_df


def valid_one_epoch_copy(valid_df, model, valid_loader, CFG):
    model.eval()
    correct = 0
    total = 0
    recall = 0
    # edge_creator = Edge_generator()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    valid_df['predict'] = ''
    for _, (images, gray_imgs, gt, ids) in pbar:
        # print(ids)
        images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        gt = gt.to(CFG.device)
        #         edges = edge_creator(gray_imgs)
        #         edges = torch.tensor(edges)
        #         edges = edges.to(CFG.device, dtype=torch.float)

        #         images += edges
        y_preds = model(images)
        prob = torch.nn.functional.softmax(y_preds)
        result = prob[:, 1].detach().cpu().numpy()

        valid_df.loc[valid_df['img_name'].isin(ids), 'predict'] = result

    preds = valid_df.to_numpy()
    labels = valid_df.to_numpy()
    tampers = labels[labels[:, 2] == '1']
    untampers = labels[labels[:, 2] == '0']

    pred_tampers = preds[np.in1d(preds[:, 1], tampers[:, 1])]
    pred_untampers = preds[np.in1d(preds[:, 1], untampers[:, 1])]

    thres = np.percentile(pred_untampers[:, -1], np.arange(90, 100, 1))
    recall = np.mean(np.greater(pred_tampers[:, -1][:, np.newaxis], thres).mean(axis=0))

    print(recall * 100)
    return recall * 100
