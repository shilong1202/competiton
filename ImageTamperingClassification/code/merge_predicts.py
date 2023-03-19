import os
from config import *
import numpy as np
import pandas as pd
import utils
import process
if __name__ == '__main__':
        col_name = ['img_name', 'img_path', 'pred_prob1','pred_prob2','pred_prob3','pred_prob4']

        imgs_info = []  # img_name, img_path, pred_prob
        test_imgs = os.listdir(test_img_path)
        test_imgs.sort(key=lambda x: x[:-4])  # sort
        for img_name in test_imgs:
            if img_name.endswith('.jpg'):  # pass other files
                imgs_info.append([img_name, os.path.join(test_img_path, img_name), 0,0,0,0])

        imgs_info_array = np.array(imgs_info)
        test_df = pd.DataFrame(imgs_info_array, columns=col_name)

        data_transforms = process.build_transforms()
        test_loader = process.build_dataloader(test_df, False, None, data_transforms)  # dataset & dtaloader

        ###############################################################
        ##### >>>>>>> step3: test <<<<<<
        ###############################################################
        model_path = [
                'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\pre_efficientnetv2_m_img512_8bs_1e4_pre\\best_fold0_epoch11.pth',
                'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\pre_efficientnetv2_m_img512_8bs_1e4_pre\\best_fold1_epoch15.pth',
                'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\pre_efficientnetv2_m_img512_8bs_1e4_pre\\best_fold2_epoch13.pth',
                'E:\softSpace\PycharmSpaces\competition\ImageTamperingDetection\output\pre_efficientnetv2_m_img512_8bs_1e4_pre\\best_fold3_epoch11.pth']
        test_df = utils.test_one_epoch(test_df, model_path, test_loader)
        submit_df = test_df.loc[:, ['img_name', 'pred_prob1','pred_prob2','pred_prob3','pred_prob4']]

        submit_df.to_csv("../output/submit/submit_318_effv2_.csv", header=False, index=False, sep=' ')