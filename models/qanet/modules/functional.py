import torch

def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)
