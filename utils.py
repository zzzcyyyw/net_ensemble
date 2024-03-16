import torch
import numpy as np

from srcs.models.modnet_ensemble_v2 import EnsembleMODNet
from srcs.models.modnet import MODNet

def pre_train_to_ensemble(ensemble_size=4, pre_train_dir="pretrained/modnet_pretrained.ckpt", ensembe_pre_train_dir="pretrained/ensemble_pretrained.pth"):
    
    # create MODNet and load the pre-trained ckpt
    ensemble_net = EnsembleMODNet(backbone_pretrained=False, ensemble_size=ensemble_size)
    #ensemble_ckpt = torch.load(ensembe_pre_train_dir, map_location="cpu")
    #ensemble_ckpt = {k.replace('module.', ''): v for k, v in ensemble_ckpt.items()}    #turn nnn.Dataparalle to cuda
    #ensemble_net.load_state_dict(ensemble_ckpt)
    
    #load the official_model
    ckpt = torch.load(pre_train_dir, map_location="cpu")
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    ensemble_net.load_state_dict(ckpt, strict=False)
    ensemble_dict = ensemble_net.state_dict()
    for k, v in ckpt.items():
        if "f_branch" in k:
            s = k.split(".")

            for i in range(ensemble_size):
                name = "f_branch_ensemble.%d.0" % (i)
                tail_name = ".".join(s[1:])
                dict_name = name + "." + tail_name
                ensemble_dict[dict_name] = v.clone()
        
    ensemble_net.load_state_dict(ensemble_dict)
    return ensemble_net


def pre_train_to_ensemble_v2(ensemble_size=4, pre_train_dir="pretrained/modnet_pretrained.ckpt", ensembe_pre_train_dir="pretrained/ensemble_pretrained.pth"):
    # create MODNet and load the pre-trained ckpt
    ensemble_net = EnsembleMODNet(backbone_pretrained=False, ensemble_size=ensemble_size)
    
    #load the official_model
    ckpt = torch.load(pre_train_dir, map_location="cpu")
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    #ensemble_net.load_state_dict(ckpt, strict=False)
    ensemble_dict = ensemble_net.state_dict()
    for k, v in ckpt.items():
        if "f_branch" in k:
            s = k.split(".")

            for i in range(1):
                name = "f_branch_ensemble.%d.0" % (i)
                tail_name = ".".join(s[1:])
                dict_name = name + "." + tail_name
                ensemble_dict[dict_name] = v.clone()
        else:
            if k in ensemble_dict:
                ensemble_dict[k] = v.clone()
                
    ensemble_net.load_state_dict(ensemble_dict)
    for k, v in ensemble_net.named_parameters():
        if "f_branch_ensemble.0.0." in k:
            v.requires_grad = False
        else:
            if k in ckpt:
                v.requires_grad = False

    return ensemble_net

def load_modnet(pre_train_dir="pretrained/modnet_pretrained.ckpt"):
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    
    #load the official_model
    ckpt = torch.load(pre_train_dir, map_location="cpu")
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    modnet.load_state_dict(ckpt)
    return modnet


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_mse(pred, alpha):
    num_pixels = float(np.prod(alpha.shape))
    return ((pred - alpha) ** 2).sum() / num_pixels


def compute_sad(pred, alpha):
    diff = np.abs(pred - alpha)
    return np.sum(diff) / 1000