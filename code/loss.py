import torch


def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()
    # CELoss = torch.nn.BCELoss()
    return {"CELoss": CELoss}


def mvss_loss():
    pass


def cat_loss():
    pass
