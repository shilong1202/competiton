import os
import utils
from config import *
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, KFold
import process
import model as mo
import loss
import time
import matplotlib.pyplot as plt


if __name__ =='__main__':
    utils.set_seed(seed)
    ckpt_path = f"../{ckpt_fold}/{ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    col_name = ['img_name', 'img_path', 'img_label']
    imgs_info = []  # img_name, img_path, img_label
    for img_name in os.listdir(tampered_img_paths):
        if img_name.endswith('.jpg'):  # pass other files
            imgs_info.append(["p_" + img_name, os.path.join(tampered_img_paths, img_name), 1])

    for img_name in os.listdir(untampered_img_paths):
        if img_name.endswith('.jpg'):  # pass other files
            imgs_info.append(["n_" + img_name, os.path.join(untampered_img_paths, img_name), 0])

    imgs_info_array = np.array(imgs_info)
    df = pd.DataFrame(imgs_info_array, columns=col_name)

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = fold

    df.to_csv(OOF_PATH, header=False, index=False, sep=' ')

    fig = plt.figure()

    for fold in range(n_fold):
        print(f'#' * 40, flush=True)
        print(f'###### Fold: {fold}', flush=True)
        print(f'#' * 40, flush=True)

        data_transforms = process.build_transforms()
        train_loader, valid_loader = process.build_dataloader(df, True, fold, data_transforms)
        model = mo.build_model(pretrain_flag=True)  # model

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)
        losses_dict = loss.build_loss()  # loss

        best_val_acc = 0
        best_epoch = 0

        acc_list=[]
        epoch_size_list =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

        score_list = []

        for epoch in range(1, epoch + 1):
            start_time = time.time()

            utils.train_one_epoch(model, train_loader, optimizer, losses_dict)
            lr_scheduler.step()
            val_acc = utils.valid_one_epoch_ture(model, valid_loader)
            acc_list.append(val_acc)

            is_best = (val_acc > best_val_acc)
            best_val_acc = max(best_val_acc, val_acc)
            if is_best:
                save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                if os.path.isfile(save_path):
                    os.remove(save_path)
                torch.save(model.state_dict(), save_path)

            epoch_time = time.time() - start_time
            print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_val_acc), flush=True)

        plt.plot(epoch_size_list, acc_list)
        plt.scatter(epoch_size_list, acc_list, c='red')
        plt.title('Training Accuracy' + ' Fold:' + str(fold))
        plt.xlabel("epoch", fontdict={'size': 16})
        plt.ylabel("acc", fontdict={'size': 16})

    plt.show()