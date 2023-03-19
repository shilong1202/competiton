import cv2
import numpy as np
import torch  # PyTorch
from torch.utils.data import Dataset, DataLoader
import albumentations as A  # Augmentations
from config import *
import timm


def build_transforms():
    data_transforms = {
        "train": A.Compose([
            A.Resize(img_size[0],img_size[1], interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),

            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=img_size[0]//20, max_width=img_size[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),

        "valid_test": A.Compose([
            A.Resize(img_size[0],img_size[1], interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            # A.HorizontalFlip(p=0.5),
            # # A.VerticalFlip(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            # A.OneOf([
            #     A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            #     # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            #     A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            # ], p=0.25),
            # A.CoarseDropout(max_holes=8, max_height=img_size[0] // 20, max_width=img_size[1] // 20,
            #                 min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0)
    }
    return data_transforms


class build_dataset(Dataset):
    def __init__(self, df, train_val_flag=True, transforms=None):

        self.df = df
        self.train_val_flag = train_val_flag  #
        self.img_paths = df['img_path'].tolist()
        self.ids = df['img_name'].tolist()
        self.transforms = transforms

        if train_val_flag:
            self.label = df['img_label'].tolist()

    def __len__(self):
        return len(self.df)
        # return 8

    def __getitem__(self, index):
        #### id
        id = self.ids[index]
        #### image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # [h, w, c]

        if self.train_val_flag:  # train
            ### augmentations
            data = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1))  # [c, h, w]
            gt = self.label[index]
            return torch.tensor(img), torch.tensor(int(gt))

        else:  # test
            ### augmentations
            data = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1))  # [c, h, w]
            return torch.tensor(img), id


def build_dataloader(df, train_val_flag=True, fold=None, data_transforms=None):
    if train_val_flag:
        train_df = df.query("fold!=@fold").reset_index(drop=True)
        valid_df = df.query("fold==@fold").reset_index(drop=True)

        train_dataset = build_dataset(train_df, train_val_flag=train_val_flag, transforms=data_transforms['train'])
        valid_dataset = build_dataset(valid_df, train_val_flag=train_val_flag, transforms=data_transforms['valid_test'])

        train_loader = DataLoader(train_dataset, batch_size=train_bs, num_workers=0, shuffle=True, pin_memory=True,
                                  drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=valid_bs, num_workers=0, shuffle=False, pin_memory=True)

        return train_loader, valid_loader

    else:
        test_dataset = build_dataset(df, train_val_flag=train_val_flag, transforms=data_transforms['valid_test'])
        test_loader = DataLoader(test_dataset, batch_size=train_bs, num_workers=0, shuffle=False, pin_memory=True,
                                 drop_last=False)  # False False
        return test_loader


def build_model(pretrain_flag=False):
    if pretrain_flag:
        pretrain_weights = "imagenet"
    else:
        pretrain_weights = False
    model = timm.create_model(backbone, pretrained=pretrain_flag, num_classes=num_classes)
    model.to(device)
    return model
