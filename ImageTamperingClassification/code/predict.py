import os
from config import *
import numpy as np
import pandas as pd
import utils
import process
if __name__ == '__main__':
        col_name = ['img_name', 'img_path', 'pred_prob']

        imgs_info = []  # img_name, img_path, pred_prob
        test_imgs = os.listdir(test_img_path)
        test_imgs.sort(key=lambda x: x[:-4])  # sort
        for img_name in test_imgs:
            if img_name.endswith('.jpg'):  # pass other files
                imgs_info.append([img_name, os.path.join(test_img_path, img_name), 0])

        imgs_info_array = np.array(imgs_info)
        test_df = pd.DataFrame(imgs_info_array, columns=col_name)

        data_transforms = process.build_transforms()
        test_loader = process.build_dataloader(test_df, False, None, data_transforms)

        ckpt_paths = [
            ""]  # please use your ckpt path
        test_df = utils.test_one_epoch(test_df, ckpt_paths, test_loader)
        submit_df = test_df.loc[:, ['img_name', 'pred_prob']]
        submit_df.to_csv("../output/", header=False, index=False, sep=' ')