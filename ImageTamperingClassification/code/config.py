import torch

seed = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_fold = "output"
ckpt_name = ""  # for submit.

test_img_path = "../input\\test\\imgs"
# tampered_img_paths = "../input\\CASIA2\\Tp"
# untampered_img_paths = "../input\\CASIA2\\Au"
tampered_img_paths = "../input\\train\\tampered\\imgs"
untampered_img_paths = "../input\\train\\untampered"

PRE_MODEL_PATH = ""


n_fold = 4
img_size = [512, 512]
train_bs = 8
valid_bs = train_bs * 2


backbone = 'efficientnetv_b4'


num_classes = 2


epoch = 20
lr = 1e-4
wd = 1e-5
lr_drop = 8

thr = 0.5