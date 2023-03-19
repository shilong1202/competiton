import timm
import torch
from torch import nn
from model_effv2 import efficientnetv2_m as create_model
from config import *


def build_model(pretrain_flag=True):
    if pretrain_flag:
        pretrain_weights = "imagenet"
    else:
        pretrain_weights = False
    # model = timm.create_model(backbone, pretrained=pretrain_flag, num_classes=num_classes)

    model = create_model(num_classes=2).to(device)
    # weights_dict = torch.load(PRE_MODEL_PATH, map_location=device)
    # load_weights_dict = {k: v for k, v in weights_dict.items()
    #                      if model.state_dict()[k].numel() == v.numel()}
    # model.load_state_dict(load_weights_dict, strict=False)
    model.to(device)
    # print(model)
    return model