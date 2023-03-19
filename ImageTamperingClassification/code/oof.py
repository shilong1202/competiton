import csv
import torch
import pandas as pd
import timm
import utils
import os
import numpy as np
import process
def read_oof(path):
    f = open(path)
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    reader = csv.reader(f)
    for row in reader:
        row = row[0]
        name, add, label, flod = row.split(' ')
        set = {"img_name": name, "img_path": add, "label": label, "fold": flod, "prob": 0}
        if flod == "0.0":
            list1.append(set)
        elif flod == "1.0":
            list2.append(set)
        elif flod == "2.0":
            list3.append(set)
        elif flod == "3.0":
            list4.append(set)
        elif flod == "4.0":
            list5.append(set)
    return pd.DataFrame(list1), pd.DataFrame(list2), pd.DataFrame(list3), pd.DataFrame(list4), pd.DataFrame(list5)


def get_prob(df,model_path):
    # col_name = ['img_name', 'img_path', 'pred_prob','label','fold']
    #
    # imgs_info = []  # img_name, img_path, pred_prob
    # test_imgs = os.listdir(test_img_path)
    # test_imgs.sort(key=lambda x: x[:-4])  # sort
    # for img_name in test_imgs:
    #     if img_name.endswith('.jpg'):  # pass other files
    #         imgs_info.append([img_name, os.path.join(test_img_path, img_name), 0])
    #
    # imgs_info_array = np.array(imgs_info)
    # test_df = pd.DataFrame(imgs_info_array, columns=col_name)

    data_transforms = process.build_transforms()
    test_loader = process.build_dataloader(df, False, None, data_transforms)  # dataset & dtaloader

    ###############################################################
    ##### >>>>>>> step3: test <<<<<<
    ###############################################################
    ckpt_paths = [model_path]  # please use your ckpt path
    test_df = utils.test_one_epoch(df, ckpt_paths, test_loader)
    submit_df = test_df.loc[:, ['img_name', 'pred_prob','fold']]
    return submit_df

if __name__ == '__main__':
    # 用df读取文件
    path = 'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\oof\oof_b4_512_8bs_2e3.csv'
    df1, df2, df3, df4, df5 = read_oof(path)

    # 定义对应的预测结果
    model_path =['E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\efficientnetb4_img512512_8bs_2e3\\best_fold0_epoch12.pth',
                 'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\efficientnetb4_img512512_8bs_2e3\\best_fold1_epoch11.pth',
                 'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\efficientnetb4_img512512_8bs_2e3\\best_fold2_epoch19.pth',
                 'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\efficientnetb4_img512512_8bs_2e3\\best_fold3_epoch16.pth',
                 'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\efficientnetb4_img512512_8bs_2e3\\best_fold4_epoch14.pth']
    df1 = get_prob(df1,model_path[0])
    df2 = get_prob(df2,model_path[1])
    df3 = get_prob(df3,model_path[2])
    df4 = get_prob(df4,model_path[3])
    df5 = get_prob(df5,model_path[4])
    df = pd.concat(df1,df2)
    df = pd.concat(df,df3)
    df = pd.concat(df,df4)
    df = pd.concat(df,df5)
    df.to_csv("../output/submit/submit_dummy_b0_768_2_tta.csv", header=False, index=False, sep=' ')
