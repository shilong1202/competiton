import os
from config import *
import csv
import numpy as np
import pandas as pd

# col_name = ['img_name', 'pred_prob1', 'pred_prob2']
# img_s_info = []
# test_img_s = os.listdir(test_img_path)
# test_img_s.sort(key=lambda x: x[:-4])  # sort
# for img_name in test_img_s:
#     if img_name.endswith('.jpg'):  # pass other files
#         img_s_info.append([img_name, 0, 0])
# df  = np.array(list)

submit_list = [
    'E:/softSpace/PycharmSpaces/competition/ImageTamperingDetection/output/submit/811.csv',
    '../output/submit/submit_318_effv2_.csv']

sub1 = open(submit_list[0])
reader1 = csv.reader(sub1)
sub1_pro = []
for row in reader1:
    row = row[0]
    name, pro2 = row.split(' ')
    sub1_pro.append(pro2)


sub2 = open(submit_list[1])
reader2 = csv.reader(sub2)
i = 0
list_pro = []
for row in reader2:
    row = row[0]
    name, pro1 ,pro2,pro3,pro4,= row.split(' ')
    pro = float(sub1_pro[i]) + (float(pro1)+float(pro3)+float(pro2)+float(pro4))/4
    i = i + 1
    set1 = {"img_name": name, "prob": str(pro/2)}
    list_pro.append(set1)

df = pd.DataFrame(list_pro)
df.to_csv("../output/submit/submit_318_effv2+811.csv", header=False, index=False, sep=' ')